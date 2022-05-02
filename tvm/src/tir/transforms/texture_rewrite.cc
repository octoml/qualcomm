#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
/**
 * This pass attempts to rewrite texture2d_load() calls to take advantage of
 * hardware OOB handling.  After ScheduleOp(), the generated IR contains the 
 * following pattern for OOB handling:
 *   tir.if_then_else(cond, tir.texture2d_load(), 0)
 * where cond decides if the coordinates in tir.texture2d_load() is OOB.
 * This pass attempts to rewrite the above pattern to the following,
 *   tir.texture2d_load(handle, tir.if_then_else(cond, row_coord, -1), ...)
 * i.e. if cond predicates that the access is OOB, we directly pass -1 as
 * x coord in tir.texture2d_load(), which automatically returns 0.
 *
 * Correspondingly in the OpenCL codegen, tir.if_then_else() will be lowered
 * to select() when lowering calls to tir.texture2d_load(). The original
 * ternary ? : would result in branching instructions whereas select() usually
 * does not.
 *
 * An alternative to this is to change the OpenCL codegen to lower
 * tir.if_then_else to select(); however, this would result in more verbose code
 * due to the ambiguity when using select().
 */
class RewriteTextureConditionalsPass : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode* op) override {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    if (op->op.same_as(builtin::if_then_else())) {
      ICHECK_EQ(op->args.size(), 3) << "if_then_else should take 3 args";
      PrimExpr cond = op->args[0];
      const CallNode* call_node = op->args[1].as<CallNode>();
      const FloatImmNode* false_val = op->args[2].as<FloatImmNode>();

      if (call_node && false_val && false_val->value == 0) {
        if (call_node->op.same_as(builtin::texture2d_load())) {
          PrimExpr handle = call_node->args[0];
          PrimExpr row_offset = call_node->args[1];
          PrimExpr col_offset = call_node->args[2];
          PrimExpr idx = call_node->args[3];

          PrimExpr row_offset_cond = Call(
              row_offset.dtype(),
              builtin::if_then_else(),
              {cond, row_offset, -1});
          PrimExpr new_texture2d_load = Call(
              call_node->dtype,
              builtin::texture2d_load(),
              {handle, row_offset_cond, col_offset, idx});
          return new_texture2d_load;
        }
      }
    }

    return expr;
  }
};

PrimFunc RewriteTextureConditionals(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = RewriteTextureConditionalsPass()(fptr->body);
  return func;
}

namespace transform {

Pass RewriteTextureConditionals() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return RewriteTextureConditionals(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveTextureConditionals", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RewriteTextureConditionals").set_body_typed(RewriteTextureConditionals);

} // namespace transform
} // namespace tir
} // namespace tvm

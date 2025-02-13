//===- LowerToBackendContract.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-lower-to-backend-contract"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/// torch-mlir的机制是，将PyTorch通过torch-script或是lazytensor转换为Torch的IR，
/// 然后将torch ir转换为合约ir。
/// 目前已支持的ir诸如tosa ir，stableHLO ir等，都支持从contraction ir转换而来。

//===----------------------------------------------------------------------===//
// Checking the backend contract.
// 类型检查: 验证所有值类型符合后端要求（如静态形状、确定的数据类型等）。
// Pass管道迭代: 多次运行优化管道，逐步消除不符合后端契约的代码模式。
// 错误处理: 在超过最大迭代次数后输出诊断信息，帮助定位问题。
//===----------------------------------------------------------------------===//

static void markDecomposedOpsAsIllegal(MLIRContext *context,
                                       ConversionTarget &target,
                                       llvm::StringSet<> backendLegalOps);

/// 检查给定类型是否符合后端契约
static LogicalResult checkType(Operation *op, Type type,
                               bool actuallyEmitDiagnostics) {
  // Allow various scalar types that backends are expected to be able to handle.
  // 基础type，后端默认能够处理
  if (isa<Torch::IntType, Torch::FloatType, Torch::BoolType, Torch::DeviceType>(
          type))
    return success();

  // 这两种类型不支持动态，允许后端进行静态匹配，用optional<>类型来表示。
  // Backends are not expected to support dynamic computations on these types,
  // but they frequently appear as parameters to ops which backends
  // can statically pattern match and eliminate from the program.
  // For example, a tensor operand might be optional, and the backend
  // will pattern-match statically whether it is passed as a tensor or None.
  if (isa<Torch::NoneType, Torch::StringType>(type))
    return success();

  // We blanket prohibit non-value-semantic tensors.
  // All of our backends are currently based on value-semantic tensors, so
  // we consider it our responsibility to lower all non-value-semantic tensors
  // to value-semantic tensors.
  // 对于non-value semantics，默认后端都是基于value semantics展开的（tensor 表示就是一个值表示）。
  // 因此到contract的时候，我们需要确保所有的tensor都是value semantics的。
  // 将non-value semantics转换为value semantics，是在torch-mlir的项目中的MaximizeValueSemantics完成的。
  if (isa<NonValueTensorType>(type)) {
    if (actuallyEmitDiagnostics) {
      return op
          ->emitError("unsupported by backend contract: non-value tensor type")
          .attachNote()
          .append("this is likely due to a missing case in the "
                  "MaximizeValueSemantics pass");
    } else {
      return failure();
    }
  }

  // 对于value semantics的tensor，我们需要确保rank和dtype是已知的。
  // 对于rank未知的tensor，我们不知道后端如何处理。
  // 因此torch-mlir认定，rank等的处理必须在codegen之前，在
  // torch-mlir内部完成。
  // For value-semantic tensors, we require at least a known rank and dtype.
  // We are not aware of a situation where our backends can handle an unranked
  // tensor type or a tensor with a dynamic dtype.
  //
  // There are somewhat fundamental reasons for this. In particular, the problem
  // of unranked codegen is completely different from the problem of ranked
  // codegen (since ranked corresponds to a fixed loop nest structure). For all
  // codegen systems we are aware of, the program must be reduced to operate
  // on ranked tensors at some point in compilation, and we are not aware of
  // any backend with a general solution to this problem before it reaches
  // codegen. So we consider it our responsibility to eliminate unranked tensor
  // from the program.
  //
  // We aren't aware of any backend with any infrastructure to represent dynamic
  // dtypes, let alone transform and optimize them. Additionally, it is unlikely
  // that any backend, even if it supports dynamic dtypes in some form, will
  // have an sufficiently rich system for representing PyTorch type promotion
  // rules. So we consider it our responsibility to ensure that all dtypes are
  // statically known.

  // torch-mlir的type system的写法是十分值得学习的！！！
  // ValueTensorType有两个属性：shape和dtype，这两个属性都是已知的。
  // 通过hasSizes()和hasDtype()来判断是否已知。
  if (auto tensorType = dyn_cast<ValueTensorType>(type)) {
    if (!tensorType.hasSizes()) {
      if (actuallyEmitDiagnostics) {
        return op
            ->emitError(
                "unsupported by backend contract: tensor with unknown rank")
            .attachNote()
            .append("this is likely due to a missing transfer function "
                    "in abstract_interp_lib_gen.py");
      } else {
        return failure();
      }
    }
    if (!tensorType.hasDtype()) {
      if (actuallyEmitDiagnostics) {
        return op
            ->emitError(
                "unsupported by backend contract: tensor with unknown dtype")
            .attachNote()
            .append("this is likely due to a missing transfer function in "
                    "abstract_interp_lib_gen.py");
      } else {
        return failure();
      }
    }
    return success();
  }

  // Optional types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary.
  if (auto optionalType = dyn_cast<OptionalType>(type)) {
    // TODO: Be stricter about tensor types.
    // See comment below for ListType.
    if (isa<ValueTensorType>(optionalType.getContainedType()))
      return success();
    return checkType(op, optionalType.getContainedType(),
                     actuallyEmitDiagnostics);
  }

  // list ops后端也是没法处理的，但是在表示pytorch code的时候有必须存在：
  // 比如卷积操作的strides就是一个list。
  // 因此需要做转换。
  // List types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary. For example, the
  // strides of a convolution op are represented as a list.
  if (auto listType = dyn_cast<ListType>(type)) {
    // TODO: Be stricter about tensor types.
    // For the moment, there are cases (such as for torch.cat) where we end
    // up with `!torch.list<vtensor>` which doesn't have shape or dtype in
    // the contained type information. Somehow this slips through and works.
    // We should be stricter about this and properly infer the contained type
    // and shape.
    // 如果list type的contained type是tensor，那么就是符合后端要求的。
    // 否则，针对list type里面存储的type，进行checktype。
    if (isa<ValueTensorType>(listType.getContainedType()))
      return success();
    return checkType(op, listType.getContainedType(), actuallyEmitDiagnostics);
  }
  // Tuple types are also in the category of types which we don't expect
  // backends to dynamically compute with, but they can be pattern matched
  // in many cases that are practically necessary.
  // 和list type一样的处理方式。
  if (auto tupleType = dyn_cast<Torch::TupleType>(type)) {
    for (auto containedType : tupleType.getContainedTypes()) {
      if (failed(checkType(op, containedType, actuallyEmitDiagnostics)))
        return failure();
    }
    return success();
  }

  // Unsupported type.
  if (actuallyEmitDiagnostics) {
    return op->emitError("unsupported by backend contract: type ") << type;
  } else {
    return failure();
  }
}

static LogicalResult checkOpIsBackendLegal(Operation *op,
                                           const ConversionTarget &target,
                                           bool actuallyEmitDiagnostics) {
  if (target.isLegal(op))
    return success();

  if (actuallyEmitDiagnostics) {
    return op->emitError("found an op that was marked as backend illegal")
        .attachNote()
        .append("this is likely due to DecomposeComplexOps being unable to "
                "decompose this op");
  } else {
    return failure();
  }
}

/// 这个pass的核心逻辑所在
/// 基于上面的checkType，对于所有的invalid type，做contract化。
static bool satisfiesBackendContract(ModuleOp module,
                                     const ConversionTarget &target,
                                     bool actuallyEmitDiagnostics = false) {
  // 查找GlobalSlotModuleInitializerOp，若存在则报错。
  // We do not permit `torch.global_slot`'s in the backend contract, since
  // support for them is not widespread, and this does not align with PyTorch's
  // more tracing-based direction.
  //
  // We just check for the GlobalSlotModuleInitializerOp since its verifier
  // ensures that the set of global slots matches those initialized by the
  // module initializer.
  auto walkResult0 = module.walk([&](Torch::GlobalSlotModuleInitializerOp op) {
    if (actuallyEmitDiagnostics) {
      // Report the error on the terminator to avoid dumping the whole
      // initializer itself, which can have pages of ops in it.
      op.getBody()
          ->getTerminator()
          ->emitError("unsupported by backend contract: module initializers")
          .attachNote()
          .append("this is likely due to InlineGlobalSlots being unable to "
                  "inline a global slot");
    }
    return WalkResult::interrupt();
  });
  if (walkResult0.wasInterrupted())
    return false;

  // Check for unimplemented operators first to give more direct diagnostics.
  walkResult0 = module.walk([&](Torch::OperatorOp op) {
    if (llvm::all_of(op.getResults(), [&op](auto res) {
          return succeeded(checkType(op.getOperation(), res.getType(),
                                     /*actuallyEmitDiagnostics=*/false));
        })) {
      return WalkResult::advance();
    }

    if (actuallyEmitDiagnostics) {
      op->emitError(
          "unsupported by backend contract: Unimplemented operator '" +
          op.getName() + "'");
    }
    return WalkResult::interrupt();
  });
  if (walkResult0.wasInterrupted())
    return false;

  // Check all the types of all Value's in the program and the legality of all
  // the ops.
  //
  // A pre-order walk gives a more intuitive "first error".
  // TODO: Should we report more than the first error?
  // How do we avoid making it too spammy?
  // 全面检查：
  // 块参数 (Block Arguments): 检查每个块的输入参数类型。
  // 操作结果 (Op Results): 检查每个操作输出结果的类型。
  // 遍历顺序: 使用PreOrder遍历，优先检查父操作，便于尽早发现错误。
  auto walkResult1 = module.walk<WalkOrder::PreOrder>([&](Block *block) {
    for (BlockArgument arg : block->getArguments())
      if (failed(checkType(block->getParentOp(), arg.getType(),
                           actuallyEmitDiagnostics))) {
        return WalkResult::interrupt();
      }

    // 这部分是该op有没有标记为legal
    for (Operation &op : *block) {
      if (failed(checkOpIsBackendLegal(&op, target, actuallyEmitDiagnostics)))
        return WalkResult::interrupt();

      for (OpResult result : op.getResults())
        if (failed(checkType(&op, result.getType(), actuallyEmitDiagnostics)))
          return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (walkResult1.wasInterrupted())
    return false;
  return true;
}

// Explicitly set ops and dialects allowed and not allowed in backend contract.
static ConversionTarget
getBackendContractTarget(MLIRContext *context, bool decompose,
                         llvm::StringSet<> backendLegalOpsSet) {
  // 设立legal dialect
  ConversionTarget target(*context);
  target.addLegalDialect<func::FuncDialect, Torch::TorchDialect>();
  if (decompose)
    // 将所有的符合操作标记为illegal
    markDecomposedOpsAsIllegal(context, target, backendLegalOpsSet);
  return target;
}

// todo：Torch-mlir debug
namespace {
class LowerToBackendContractPass
    : public LowerToBackendContractBase<LowerToBackendContractPass> {
public:
  LowerToBackendContractPass() = default;
  LowerToBackendContractPass(int maxIterations, bool decompose,
                             bool shapeDtypeRefine,
                             ArrayRef<std::string> backendLegalOps,
                             StringRef extraLibrary) {
    this->maxIterations = maxIterations;
    this->decompose = decompose;
    this->shapeDtypeRefine = shapeDtypeRefine;
    this->backendLegalOps = backendLegalOps;
    this->extraLibrary = extraLibrary.str();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    backendLegalOpsSet.clear();
    backendLegalOpsSet.insert(backendLegalOps.begin(), backendLegalOps.end());
    ConversionTarget target =
        getBackendContractTarget(context, decompose, backendLegalOpsSet);

    OpPassManager pm(module.getOperationName());
    TorchLoweringPipelineOptions options;
    options.decompose = decompose;
    options.shapeDtypeRefine = shapeDtypeRefine;
    options.backendLegalOps = backendLegalOps;
    options.extraLibrary = extraLibrary;

    // 这个pm pipeline使整个contract 化的核心。
    // Torch-mlir debug here:
    /// Creates a pipeline that simplifies the computations in the program.
    /// This pass does not do any global program restructuring -- it works entirely
    /// within a single semantic model of a `builtin.module` with
    /// `torch.global_slot` ops and `func.func` ops.
    createTorchSimplificationPipeline(pm, options);

    // 多轮迭代，直到全部contract化
    int i = 0;
    do {
      if (i++ == maxIterations) {
        LLVM_DEBUG({
          llvm::dbgs() << "LowerToBackendContractPass: "
                       << "failed to satisfy backend contract after "
                       << maxIterations
                       << " iterations of the simplification pipeline\n";
        });
        // Show the diagnostics.
        (void)satisfiesBackendContract(module, target,
                                       /*actuallyEmitDiagnostics=*/true);
        return signalPassFailure();
      }

      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    } while (!satisfiesBackendContract(module, target));
    LLVM_DEBUG({
      llvm::dbgs() << "LowerToBackendContractPass: " << "succeeded after " << i
                   << " iterations of the simplification pipeline\n";
    });
  }

private:
  llvm::StringSet<> backendLegalOpsSet;
};

class VerifyBackendContractNoDecompositionsPass
    : public VerifyBackendContractNoDecompositionsBase<
          VerifyBackendContractNoDecompositionsPass> {
public:
  VerifyBackendContractNoDecompositionsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target =
        getBackendContractTarget(context, /*decompose*/ false,
                                 /*backendLegalOpsSet*/ {});

    if (!satisfiesBackendContract(getOperation(), target,
                                  /*actuallyEmitDiagnostics=*/true)) {
      return signalPassFailure();
    }
  }
};
} // namespace

// 十分简洁的c++ code写法
// 工厂方法。
std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createLowerToBackendContractPass(
    int maxIterations, bool decompose, bool shapeDtypeRefine,
    ArrayRef<std::string> backendLegalOps, StringRef extraLibrary) {
  return std::make_unique<LowerToBackendContractPass>(
      maxIterations, decompose, shapeDtypeRefine, backendLegalOps,
      extraLibrary);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createVerifyBackendContractNoDecompositionsPass() {
  return std::make_unique<VerifyBackendContractNoDecompositionsPass>();
}

// 用于标记 MLIR 计算图 中 需要分解（decompose）的算子 为 非法（illegal），
// 以确保它们在 LowerToBackendContractPass 过程中被正确处理。
// The backend contract guarantees that ops with decompositions available will
// be decomposed. The only way to have an op reach the backend contract without
// getting decomposed is by having the user explicitly specify that op in the
// `backendLegalOpsSet` argument to the `LowerToBackendContractPass`. Therefore,
// here we mark as illegal all ops with decompositions except for those in
// `backendLegalOpsSet`.
//
// The legality check takes place here instead of in the `DecomposeComplexOps`
// pass for two reasons:
// 1. Makes sure the `DecomposeComplexOps` pass always succeeds, allowing it to
//   run multiple times. This is needed for graphs where static information such
//   as dtypes and shapes takes multiple iterations to propagate through the
//   entire graph. `DecomposeComplexOps` pass failing would cause the entire
//   `LowerToBackendContractPass` to fail
// 2. Makes the legality requirements in the backend contract for ops with
//   decompositions explicit in this file
static void markDecomposedOpsAsIllegal(MLIRContext *context,
                                       ConversionTarget &target,
                                       llvm::StringSet<> backendLegalOpsSet) {
  target.addIllegalOp<AtenSoftmaxIntOp>();
  target.addIllegalOp<Aten_SoftmaxOp>();
  target.addIllegalOp<Aten_SafeSoftmaxOp>();
  target.addIllegalOp<Aten_LogSoftmaxOp>();
  target.addIllegalOp<AtenLogSoftmaxIntOp>();
  target.addIllegalOp<AtenLogSigmoidOp>();
  target.addIllegalOp<AtenHardshrinkOp>();
  target.addIllegalOp<AtenSoftshrinkOp>();
  target.addIllegalOp<AtenEmptyLikeOp>();
  target.addIllegalOp<AtenOnesLikeOp>();
  target.addIllegalOp<AtenZerosLikeOp>();
  target.addIllegalOp<AtenStackOp>();
  target.addIllegalOp<AtenHstackOp>();
  target.addIllegalOp<AtenColumnStackOp>();
  target.addIllegalOp<AtenRollOp>();
  target.addIllegalOp<AtenRepeatOp>();
  target.addIllegalOp<AtenRepeatInterleaveSelfIntOp>();
  target.addIllegalOp<AtenExpandOp>();
  target.addIllegalOp<AtenFlattenUsingIntsOp>();
  target.addIllegalOp<AtenWhereScalarOp>();
  target.addIllegalOp<AtenWhereScalarOtherOp>();
  target.addIllegalOp<AtenWhereScalarSelfOp>();
  target.addIllegalOp<AtenMaskedFillScalarOp>();
  target.addIllegalOp<AtenMaskedFillTensorOp>();
  target.addIllegalOp<AtenMaskedScatterOp>();
  target.addIllegalOp<AtenSizeOp>();
  target.addIllegalOp<AtenReshapeOp>();
  target.addIllegalOp<Aten_SoftmaxBackwardDataOp>();
  target.addIllegalOp<AtenTanhBackwardOp>();
  target.addIllegalOp<AtenAtleast1dOp>();
  target.addIllegalOp<AtenAtleast2dOp>();
  target.addIllegalOp<AtenEinsumOp>();
  target.addIllegalOp<Aten_TrilinearOp>();
  target.addIllegalOp<AtenTraceOp>();
  target.addIllegalOp<AtenAddmmOp>();
  target.addIllegalOp<AtenMeanOp>();
  target.addIllegalOp<AtenMeanDimOp>();
  target.addIllegalOp<AtenNormScalarOptDimOp>();
  target.addIllegalOp<AtenSelectIntOp>();
  target.addIllegalOp<AtenMvOp>();
  target.addIllegalOp<AtenRenormOp>();
  target.addIllegalOp<AtenRot90Op>();
  target.addIllegalOp<AtenLinalgCrossOp>();
  target.addIllegalOp<Aten_LinalgDetOp>();
  target.addIllegalOp<AtenLinalgSlogdetOp>();
  target.addIllegalOp<AtenPixelShuffleOp>();
  target.addIllegalOp<AtenTOp>();
  target.addIllegalOp<Aten_LogSoftmaxBackwardDataOp>();
  target.addDynamicallyLegalOp<AtenMatmulOp>([](AtenMatmulOp op) {
    std::optional<unsigned> lhsRank = getTensorRank(op.getSelf());
    std::optional<unsigned> rhsRank = getTensorRank(op.getOther());
    if (!lhsRank || !rhsRank)
      return false;
    // Make aten.matmul legal if the following condition is satisfied.
    return (*lhsRank != 2 || *rhsRank != 2) && (*lhsRank != 3 || *rhsRank != 3);
  });
  target.addIllegalOp<AtenAddcmulOp>();
  target.addIllegalOp<AtenAddcdivOp>();
  target.addIllegalOp<Aten_WeightNormInterfaceOp>();
  target.addIllegalOp<AtenInstanceNormOp>();
  target.addIllegalOp<AtenLayerNormOp>();
  target.addIllegalOp<AtenNativeLayerNormOp>();
  target.addIllegalOp<AtenGroupNormOp>();
  target.addIllegalOp<AtenNativeGroupNormOp>();
  target.addIllegalOp<AtenNativeBatchNormOp>();
  target.addIllegalOp<Aten_ConvolutionOp, Aten_ConvolutionDeprecatedOp>();
  target.addIllegalOp<AtenConvolutionBackwardOp>();
  target.addIllegalOp<AtenConvTbcOp>();
  target.addIllegalOp<AtenConv1dOp>();
  target.addIllegalOp<AtenConv2dOp>();
  target.addIllegalOp<AtenConv3dOp>();
  target.addIllegalOp<AtenConvTranspose1dOp>();
  target.addIllegalOp<AtenConvTranspose2dInputOp>();
  target.addIllegalOp<AtenConvTranspose3dInputOp>();
  target.addIllegalOp<AtenArangeOp>();
  target.addIllegalOp<AtenArangeStartOp>();
  target.addIllegalOp<AtenLinspaceOp>();
  target.addIllegalOp<AtenArgmaxOp>();
  target.addIllegalOp<AtenArgminOp>();
  target.addIllegalOp<AtenAminmaxOp>();
  target.addIllegalOp<AtenAmaxOp>();
  target.addIllegalOp<AtenAminOp>();
  target.addIllegalOp<AtenSquareOp>();
  target.addIllegalOp<AtenVarOp>();
  target.addIllegalOp<AtenStdOp>();
  target.addIllegalOp<Aten_UnsafeViewOp>();
  target.addIllegalOp<Aten_ReshapeAliasOp>();
  target.addIllegalOp<AtenBernoulliOp>();
  target.addIllegalOp<ValsemVariantAtenBernoulliFloatOp>();
  target.addIllegalOp<AtenBernoulliPOp>();
  target.addIllegalOp<AtenBernoulliTensorOp>();
  target.addIllegalOp<AtenExponentialOp>();
  target.addIllegalOp<AtenZeroOp>();
  target.addIllegalOp<AtenEyeOp>();
  target.addIllegalOp<AtenEyeMOp>();
  target.addIllegalOp<AtenNanToNumOp>();
  target.addIllegalOp<AtenIsnanOp>();
  target.addIllegalOp<AtenIsinfOp>();
  target.addIllegalOp<AtenIsneginfOp>();
  target.addIllegalOp<AtenIsposinfOp>();
  target.addIllegalOp<AtenRandLikeOp>();
  target.addIllegalOp<AtenHardsigmoidOp>();
  target.addIllegalOp<AtenRelu6Op>();
  target.addIllegalOp<AtenEluOp>();
  target.addIllegalOp<AtenFakeQuantizePerTensorAffineOp>();
  target.addIllegalOp<AtenFakeQuantizePerTensorAffineCachemaskOp>();
  target.addIllegalOp<AtenGluOp>();
  target.addIllegalOp<AtenSeluOp>();
  target.addIllegalOp<AtenHardswishOp>();
  target.addIllegalOp<AtenSoftplusOp>();
  target.addIllegalOp<AtenSiluOp>();
  target.addIllegalOp<AtenNewZerosOp>();
  target.addIllegalOp<AtenNewOnesOp>();
  target.addIllegalOp<AtenHardtanhOp>();
  target.addIllegalOp<AtenFullOp>();
  target.addIllegalOp<AtenLinearOp>();
  target.addIllegalOp<AtenMishOp>();
  target.addIllegalOp<AtenFullLikeOp>();
  target.addIllegalOp<AtenNewFullOp>();
  target.addIllegalOp<AtenExpandAsOp>();
  target.addIllegalOp<Aten_ToCopyOp>();
  target.addIllegalOp<AtenDropoutOp>();
  target.addIllegalOp<AtenNativeDropoutOp>();
  target.addIllegalOp<AtenNewEmptyOp>();
  target.addIllegalOp<AtenIndexTensorOp>();
  target.addIllegalOp<AtenIndexPutOp>();
  target.addIllegalOp<Aten_IndexPutImplOp>();
  target.addIllegalOp<Aten_UnsafeIndexPutHackedTwinOp>();
  target.addIllegalOp<AtenPadOp>();
  target.addIllegalOp<AtenPreluOp>();
  target.addIllegalOp<AtenRreluOp>();
  target.addIllegalOp<AtenRreluWithNoiseOp>();
  target.addIllegalOp<AtenRreluWithNoiseFunctionalOp>();
  target.addIllegalOp<AtenRreluWithNoiseBackwardOp>();
  target.addIllegalOp<AtenCeluOp>();
  target.addIllegalOp<AtenToDtypeLayoutOp>();
  target.addIllegalOp<AtenToDeviceOp>();
  target.addIllegalOp<AtenToPrimDeviceOp>();
  target.addIllegalOp<AtenAdaptiveAvgPool1dOp>();
  target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
  target.addIllegalOp<AtenClampMinOp>();
  target.addIllegalOp<AtenClampMinTensorOp>();
  target.addIllegalOp<AtenClampMaxOp>();
  target.addIllegalOp<AtenBaddbmmOp>();
  target.addIllegalOp<AtenFloorDivideOp>();
  target.addIllegalOp<AtenFloorDivideScalarOp>();
  target.addIllegalOp<AtenNumpyTOp>();
  target.addIllegalOp<AtenSelectScatterOp>();
  target.addIllegalOp<AtenVarDimOp>();
  target.addIllegalOp<AtenVarCorrectionOp>();
  target.addIllegalOp<AtenStdDimOp>();
  target.addIllegalOp<AtenStdCorrectionOp>();
  target.addIllegalOp<AtenNarrowOp>();
  target.addIllegalOp<AtenNarrowTensorOp>();
  target.addIllegalOp<Aten_EmbeddingBagOp>();
  target.addIllegalOp<AtenLiftFreshCopyOp>();
  target.addIllegalOp<AtenLerpScalarOp>();
  target.addIllegalOp<AtenLerpTensorOp>();
  target.addIllegalOp<AtenMseLossOp>();
  target.addIllegalOp<AtenL1LossOp>();
  target.addIllegalOp<AtenRandintLowOp>();
  target.addIllegalOp<AtenRandintOp>();
  target.addIllegalOp<AtenVarMeanCorrectionOp>();
  target.addIllegalOp<PrimsConvertElementTypeOp>();
  target.addIllegalOp<PrimsVarOp>();
  target.addIllegalOp<PrimsSqrtOp>();
  target.addIllegalOp<AtenRandOp>();
  target.addIllegalOp<AtenRandnOp>();
  target.addIllegalOp<AtenRandnGeneratorOp>();
  target.addIllegalOp<AtenRandnLikeOp>();
  target.addIllegalOp<AtenNormalFunctionalOp>();
  target.addIllegalOp<AtenVarMeanOp>();
  target.addIllegalOp<AtenRad2degOp>();
  target.addIllegalOp<AtenCosineSimilarityOp>();
  target.addIllegalOp<AtenTruncOp>();
  target.addIllegalOp<AtenSignbitOp>();
  target.addIllegalOp<AtenFracOp>();
  target.addIllegalOp<AtenCopysignTensorOp>();
  target.addIllegalOp<AtenLdexpTensorOp>();
  target.addIllegalOp<AtenNewEmptyStridedOp>();
  target.addIllegalOp<AtenEmptyStridedOp>();
  target.addIllegalOp<AtenBucketizeTensorOp>();
  target.addIllegalOp<PrimsSqueezeOp>();
  target.addIllegalOp<AtenMovedimIntOp>();
  target.addIllegalOp<AtenOneHotOp>();
  target.addIllegalOp<AtenCrossEntropyLossOp>();
  target.addIllegalOp<AtenVarMeanDimOp>();
  target.addIllegalOp<AtenTopkOp>();
  target.addIllegalOp<AtenHannWindowPeriodicOp>();
  target.addIllegalOp<AtenScalarTensorOp>();
  target.addIllegalOp<AtenScatterValueOp>();
  target.addIllegalOp<AtenTypeAsOp>();
  target.addIllegalOp<AtenTileOp>();
  target.addIllegalOp<AtenReshapeAsOp>();
  target.addIllegalOp<AtenTriuOp>();
  target.addIllegalOp<AtenTriuIndicesOp>();
  target.addIllegalOp<AtenTrilIndicesOp>();
  target.addIllegalOp<AtenDeg2radOp>();
  target.addIllegalOp<AtenLinalgNormOp>();
  target.addIllegalOp<AtenFminOp>();
  target.addIllegalOp<AtenFmaxOp>();
  target.addIllegalOp<AtenSpecialExpm1Op>();

  // 这个backendLegalOpsSet是用户显示指定的合法Ops。
  for (auto &opName : backendLegalOpsSet) {
    target.addLegalOp(
        OperationName(kTorchOpPrefix + opName.first().str(), context));
  }
  target.addDynamicallyLegalOp<OperatorOp>(
      [backendLegalOpsSet](OperatorOp opOp) {
        auto opName = cast<StringAttr>(opOp->getAttr("name")).getValue();
        return backendLegalOpsSet.contains(opName);
      });
}

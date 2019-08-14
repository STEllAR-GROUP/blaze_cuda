//=================================================================================================
/*!
//  \file blaze_cuda/math/DenseMatrix.h
//  \brief Header file for all basic DenseMatrix functionality
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//  Copyright (C) 2019 Jules Penuchot - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_CUDA_MATH_DENSEMATRIX_H_
#define _BLAZE_CUDA_MATH_DENSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

//#include <blaze_cuda/math/adaptors/DiagonalMatrix.h>
//#include <blaze_cuda/math/adaptors/HermitianMatrix.h>
//#include <blaze_cuda/math/adaptors/LowerMatrix.h>
//#include <blaze_cuda/math/adaptors/SymmetricMatrix.h>
//#include <blaze_cuda/math/adaptors/UpperMatrix.h>
//#include <blaze_cuda/math/dense/DenseMatrix.h>
//#include <blaze_cuda/math/dense/Eigen.h>
//#include <blaze_cuda/math/dense/Inversion.h>
//#include <blaze_cuda/math/dense/LLH.h>
//#include <blaze_cuda/math/dense/LQ.h>
//#include <blaze_cuda/math/dense/LU.h>
//#include <blaze_cuda/math/dense/QL.h>
//#include <blaze_cuda/math/dense/QR.h>
//#include <blaze_cuda/math/dense/RQ.h>
//#include <blaze_cuda/math/dense/SVD.h>
#include <blaze_cuda/math/expressions/DMatDeclDiagExpr.h>
#include <blaze_cuda/math/expressions/DMatDeclHermExpr.h>
#include <blaze_cuda/math/expressions/DMatDeclLowExpr.h>
#include <blaze_cuda/math/expressions/DMatDeclSymExpr.h>
#include <blaze_cuda/math/expressions/DMatDeclUppExpr.h>
#include <blaze_cuda/math/expressions/DMatDetExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatAddExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatEqualExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatKronExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatMapExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatMultExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatSchurExpr.h>
#include <blaze_cuda/math/expressions/DMatDMatSubExpr.h>
#include <blaze_cuda/math/expressions/DMatDVecMultExpr.h>
#include <blaze_cuda/math/expressions/DMatEvalExpr.h>
#include <blaze_cuda/math/expressions/DMatInvExpr.h>
#include <blaze_cuda/math/expressions/DMatMapExpr.h>
#include <blaze_cuda/math/expressions/DMatMeanExpr.h>
#include <blaze_cuda/math/expressions/DMatNormExpr.h>
#include <blaze_cuda/math/expressions/DMatReduceExpr.h>
#include <blaze_cuda/math/expressions/DMatScalarDivExpr.h>
#include <blaze_cuda/math/expressions/DMatScalarMultExpr.h>
#include <blaze_cuda/math/expressions/DMatSerialExpr.h>
#include <blaze_cuda/math/expressions/DMatSoftmaxExpr.h>
#include <blaze_cuda/math/expressions/DMatStdDevExpr.h>
#include <blaze_cuda/math/expressions/DMatSVecMultExpr.h>
#include <blaze_cuda/math/expressions/DMatTDMatAddExpr.h>
#include <blaze_cuda/math/expressions/DMatTDMatMapExpr.h>
#include <blaze_cuda/math/expressions/DMatTDMatMultExpr.h>
#include <blaze_cuda/math/expressions/DMatTDMatSchurExpr.h>
#include <blaze_cuda/math/expressions/DMatTDMatSubExpr.h>
#include <blaze_cuda/math/expressions/DMatTransExpr.h>
#include <blaze_cuda/math/expressions/DMatVarExpr.h>
#include <blaze_cuda/math/expressions/DVecDVecOuterExpr.h>
#include <blaze_cuda/math/expressions/TDMatDMatMultExpr.h>
#include <blaze_cuda/math/expressions/TDMatDVecMultExpr.h>
#include <blaze_cuda/math/expressions/TDMatSVecMultExpr.h>
#include <blaze_cuda/math/expressions/TDMatTDMatMultExpr.h>
#include <blaze_cuda/math/expressions/TDVecDMatMultExpr.h>
#include <blaze_cuda/math/expressions/TDVecTDMatMultExpr.h>
#include <blaze_cuda/math/expressions/TSVecDMatMultExpr.h>
#include <blaze_cuda/math/expressions/TSVecTDMatMultExpr.h>
//#include <blaze_cuda/math/Matrix.h>

#endif

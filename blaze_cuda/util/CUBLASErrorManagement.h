//=================================================================================================
/*!
//  \file blaze/util/CUBLASErrorManagement.h
//  \brief Header file for CUDAReduce's implementation
//
//  Copyright (C) 2012-2019 Jules P�nuchot - All Rights Reserved
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

#ifndef _BLAZE_CUDA_UTIL_CUBLASERRORMANAGEMENT_H_
#define _BLAZE_CUDA_UTIL_CUBLASERRORMANAGEMENT_H_

#include <sstream>
#include <stdexcept>

#include <cublas_v2.h>

namespace blaze {

std::string cublasStatusToString( cublasStatus_t const& status ) {
   if ( status == CUBLAS_STATUS_SUCCESS )           return "CUBLAS_STATUS_SUCCESS";
   if ( status == CUBLAS_STATUS_NOT_INITIALIZED )   return "CUBLAS_STATUS_NOT_INITIALIZED";
   if ( status == CUBLAS_STATUS_ALLOC_FAILED )      return "CUBLAS_STATUS_ALLOC_FAILED";
   if ( status == CUBLAS_STATUS_INVALID_VALUE )     return "CUBLAS_STATUS_INVALID_VALUE";
   if ( status == CUBLAS_STATUS_ARCH_MISMATCH )     return "CUBLAS_STATUS_ARCH_MISMATCH";
   if ( status == CUBLAS_STATUS_MAPPING_ERROR )     return "CUBLAS_STATUS_MAPPING_ERROR";
   if ( status == CUBLAS_STATUS_EXECUTION_FAILED )  return "CUBLAS_STATUS_EXECUTION_FAILED";
   if ( status == CUBLAS_STATUS_INTERNAL_ERROR )    return "CUBLAS_STATUS_INTERNAL_ERROR";
   if ( status == CUBLAS_STATUS_NOT_SUPPORTED )     return "CUBLAS_STATUS_NOT_SUPPORTED";
   throw std::runtime_error("unknown");
}

}  // namespace blaze

#define CUBLAS_ERROR_CHECK(status) \
if( (status) != CUBLAS_STATUS_SUCCESS ) \
{ \
   std::stringstream ss; \
   ss << "cuBLAS error at: " << __FILE__ << ':' << __LINE__ \
      << ", in function: " << __func__ << ". CUBLAS (status): " \
      << blaze::cublasStatusToString( (status) ); \
   throw std::runtime_error( ss.str() ); \
}


#endif

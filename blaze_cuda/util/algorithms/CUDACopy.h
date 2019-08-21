//=================================================================================================
/*!
//  \file blaze_cuda/util/CUDACopy.h
//  \brief Header file for the IsCUDAEnabled type trait
//
//  Copyright (C) 2019 Jules Pénuchot - All Rights Reserved
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

#ifndef _BLAZE_CUDA_UTIL_ALGORITHMS_CUDACOPY_H_
#define _BLAZE_CUDA_UTIL_ALGORITHMS_CUDACOPY_H_

#include <cstddef>

#include <blaze/system/HostDevice.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>
#include <blaze_cuda/util/CUDAErrorManagement.h>

namespace blaze {
   template< std::size_t Unroll = 4, typename InputIt, typename OutputIt >
   inline void cuda_copy( InputIt in_begin, InputIt in_end, OutputIt out_begin )
   {
      cuda_transform( in_begin , in_end, out_begin
         , [] BLAZE_DEVICE_CALLABLE ( auto const& e ) { return e; } );
   }
}

#endif

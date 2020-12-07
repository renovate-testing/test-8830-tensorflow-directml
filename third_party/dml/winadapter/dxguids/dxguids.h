// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

constexpr inline bool ConstexprIsEqualGUID(REFGUID a, REFGUID b)
{
    return a.Data1 == b.Data1 &&
        a.Data2 == b.Data2 &&
        a.Data3 == b.Data3 &&
        a.Data4[0] == b.Data4[0] &&
        a.Data4[1] == b.Data4[1] &&
        a.Data4[2] == b.Data4[2] &&
        a.Data4[3] == b.Data4[3] &&
        a.Data4[4] == b.Data4[4] &&
        a.Data4[5] == b.Data4[5] &&
        a.Data4[6] == b.Data4[6] &&
        a.Data4[7] == b.Data4[7];
}

// Each COM interface (e.g. ID3D12Device) has a unique interface ID (IID) associated with it. With MSVC, the IID is defined 
// along with the interface declaration using compiler intrinsics (__declspec(uuid(...)); the IID can then be retrieved 
// using __uuidof. These intrinsics are not supported with all toolchains, so these helpers redefine IID values that can be 
// used with the various adapter COM helpers (ComPtr, IID_PPV_ARGS, etc.) for Linux. IIDs are stable and cannot change, but as 
// a precaution we statically assert the values are as expected when compiling for Windows.
#ifdef _WIN32
// winadapter.h isn't included when building for Windows, so the base function template needs to be declared.
template <typename T> GUID uuidof() = delete;
#define WINADAPTER_IID(InterfaceName, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
template <> constexpr GUID uuidof<InterfaceName>() \
{ \
    return { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }; \
} \
static_assert(ConstexprIsEqualGUID(uuidof<InterfaceName>(), __uuidof(InterfaceName)), "GUID definition mismatch: "#InterfaceName);
#else
#define WINADAPTER_IID(InterfaceName, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
template <> constexpr GUID uuidof<InterfaceName>() \
{ \
    return { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }; \
}
#endif

// Direct3D
WINADAPTER_IID(ID3D12Object, 0xc4fec28f, 0x7966, 0x4e95, 0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8);
WINADAPTER_IID(ID3D12DeviceChild, 0x905db94b, 0xa00c, 0x4140, 0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57);
WINADAPTER_IID(ID3D12RootSignature, 0xc54a6b66, 0x72df, 0x4ee8, 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14);
WINADAPTER_IID(ID3D12RootSignatureDeserializer, 0x34AB647B, 0x3CC8, 0x46AC, 0x84, 0x1B, 0xC0, 0x96, 0x56, 0x45, 0xC0, 0x46);
WINADAPTER_IID(ID3D12VersionedRootSignatureDeserializer, 0x7F91CE67, 0x090C, 0x4BB7, 0xB7, 0x8E, 0xED, 0x8F, 0xF2, 0xE3, 0x1D, 0xA0);
WINADAPTER_IID(ID3D12Pageable, 0x63ee58fb, 0x1268, 0x4835, 0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6);
WINADAPTER_IID(ID3D12Heap, 0x6b3b2502, 0x6e51, 0x45b3, 0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3);
WINADAPTER_IID(ID3D12Resource, 0x696442be, 0xa72e, 0x4059, 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad);
WINADAPTER_IID(ID3D12CommandAllocator, 0x6102dee4, 0xaf59, 0x4b09, 0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24);
WINADAPTER_IID(ID3D12Fence, 0x0a753dcf, 0xc4d8, 0x4b91, 0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76);
WINADAPTER_IID(ID3D12Fence1, 0x433685fe, 0xe22b, 0x4ca0, 0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a);
WINADAPTER_IID(ID3D12PipelineState, 0x765a30f3, 0xf624, 0x4c6f, 0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45);
WINADAPTER_IID(ID3D12DescriptorHeap, 0x8efb471d, 0x616c, 0x4f49, 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51);
WINADAPTER_IID(ID3D12QueryHeap, 0x0d9658ae, 0xed45, 0x469e, 0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4);
WINADAPTER_IID(ID3D12CommandSignature, 0xc36a797c, 0xec80, 0x4f0a, 0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1);
WINADAPTER_IID(ID3D12CommandList, 0x7116d91c, 0xe7e4, 0x47ce, 0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5);
WINADAPTER_IID(ID3D12GraphicsCommandList, 0x5b160d0f, 0xac1b, 0x4185, 0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55);
WINADAPTER_IID(ID3D12GraphicsCommandList1, 0x553103fb, 0x1fe7, 0x4557, 0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7);
WINADAPTER_IID(ID3D12GraphicsCommandList2, 0x38C3E585, 0xFF17, 0x412C, 0x91, 0x50, 0x4F, 0xC6, 0xF9, 0xD7, 0x2A, 0x28);
WINADAPTER_IID(ID3D12CommandQueue, 0x0ec870a6, 0x5d7e, 0x4c22, 0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed);
WINADAPTER_IID(ID3D12Device, 0x189819f1, 0x1db6, 0x4b57, 0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7);
WINADAPTER_IID(ID3D12PipelineLibrary, 0xc64226a8, 0x9201, 0x46af, 0xb4, 0xcc, 0x53, 0xfb, 0x9f, 0xf7, 0x41, 0x4f);
WINADAPTER_IID(ID3D12PipelineLibrary1, 0x80eabf42, 0x2568, 0x4e5e, 0xbd, 0x82, 0xc3, 0x7f, 0x86, 0x96, 0x1d, 0xc3);
WINADAPTER_IID(ID3D12Device1, 0x77acce80, 0x638e, 0x4e65, 0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e);
WINADAPTER_IID(ID3D12Device2, 0x30baa41e, 0xb15b, 0x475c, 0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28);
WINADAPTER_IID(ID3D12Device3, 0x81dadc15, 0x2bad, 0x4392, 0x93, 0xc5, 0x10, 0x13, 0x45, 0xc4, 0xaa, 0x98);
WINADAPTER_IID(ID3D12ProtectedSession, 0xA1533D18, 0x0AC1, 0x4084, 0x85, 0xB9, 0x89, 0xA9, 0x61, 0x16, 0x80, 0x6B);
WINADAPTER_IID(ID3D12ProtectedResourceSession, 0x6CD696F4, 0xF289, 0x40CC, 0x80, 0x91, 0x5A, 0x6C, 0x0A, 0x09, 0x9C, 0x3D);
WINADAPTER_IID(ID3D12Device4, 0xe865df17, 0xa9ee, 0x46f9, 0xa4, 0x63, 0x30, 0x98, 0x31, 0x5a, 0xa2, 0xe5);
WINADAPTER_IID(ID3D12LifetimeOwner, 0xe667af9f, 0xcd56, 0x4f46, 0x83, 0xce, 0x03, 0x2e, 0x59, 0x5d, 0x70, 0xa8);
WINADAPTER_IID(ID3D12SwapChainAssistant, 0xf1df64b6, 0x57fd, 0x49cd, 0x88, 0x07, 0xc0, 0xeb, 0x88, 0xb4, 0x5c, 0x8f);
WINADAPTER_IID(ID3D12LifetimeTracker, 0x3fd03d36, 0x4eb1, 0x424a, 0xa5, 0x82, 0x49, 0x4e, 0xcb, 0x8b, 0xa8, 0x13);
WINADAPTER_IID(ID3D12StateObject, 0x47016943, 0xfca8, 0x4594, 0x93, 0xea, 0xaf, 0x25, 0x8b, 0x55, 0x34, 0x6d);
WINADAPTER_IID(ID3D12StateObjectProperties, 0xde5fa827, 0x9bf9, 0x4f26, 0x89, 0xff, 0xd7, 0xf5, 0x6f, 0xde, 0x38, 0x60);
WINADAPTER_IID(ID3D12Device5, 0x8b4f173b, 0x2fea, 0x4b80, 0x8f, 0x58, 0x43, 0x07, 0x19, 0x1a, 0xb9, 0x5d);
WINADAPTER_IID(ID3D12DeviceRemovedExtendedDataSettings, 0x82BC481C, 0x6B9B, 0x4030, 0xAE, 0xDB, 0x7E, 0xE3, 0xD1, 0xDF, 0x1E, 0x63);
WINADAPTER_IID(ID3D12DeviceRemovedExtendedDataSettings1, 0xDBD5AE51, 0x3317, 0x4F0A, 0xAD, 0xF9, 0x1D, 0x7C, 0xED, 0xCA, 0xAE, 0x0B);
WINADAPTER_IID(ID3D12DeviceRemovedExtendedData, 0x98931D33, 0x5AE8, 0x4791, 0xAA, 0x3C, 0x1A, 0x73, 0xA2, 0x93, 0x4E, 0x71);
WINADAPTER_IID(ID3D12DeviceRemovedExtendedData1, 0x9727A022, 0xCF1D, 0x4DDA, 0x9E, 0xBA, 0xEF, 0xFA, 0x65, 0x3F, 0xC5, 0x06);
WINADAPTER_IID(ID3D12Device6, 0xc70b221b, 0x40e4, 0x4a17, 0x89, 0xaf, 0x02, 0x5a, 0x07, 0x27, 0xa6, 0xdc);
WINADAPTER_IID(ID3D12ProtectedResourceSession1, 0xD6F12DD6, 0x76FB, 0x406E, 0x89, 0x61, 0x42, 0x96, 0xEE, 0xFC, 0x04, 0x09);
WINADAPTER_IID(ID3D12Device7, 0x5c014b53, 0x68a1, 0x4b9b, 0x8b, 0xd1, 0xdd, 0x60, 0x46, 0xb9, 0x35, 0x8b);
WINADAPTER_IID(ID3D12Device8, 0x9218E6BB, 0xF944, 0x4F7E, 0xA7, 0x5C, 0xB1, 0xB2, 0xC7, 0xB7, 0x01, 0xF3);
WINADAPTER_IID(ID3D12Resource1, 0x9D5E227A, 0x4430, 0x4161, 0x88, 0xB3, 0x3E, 0xCA, 0x6B, 0xB1, 0x6E, 0x19);
WINADAPTER_IID(ID3D12Resource2, 0xBE36EC3B, 0xEA85, 0x4AEB, 0xA4, 0x5A, 0xE9, 0xD7, 0x64, 0x04, 0xA4, 0x95);
WINADAPTER_IID(ID3D12Heap1, 0x572F7389, 0x2168, 0x49E3, 0x96, 0x93, 0xD6, 0xDF, 0x58, 0x71, 0xBF, 0x6D);
WINADAPTER_IID(ID3D12GraphicsCommandList3, 0x6FDA83A7, 0xB84C, 0x4E38, 0x9A, 0xC8, 0xC7, 0xBD, 0x22, 0x01, 0x6B, 0x3D);
WINADAPTER_IID(ID3D12MetaCommand, 0xDBB84C27, 0x36CE, 0x4FC9, 0xB8, 0x01, 0xF0, 0x48, 0xC4, 0x6A, 0xC5, 0x70);
WINADAPTER_IID(ID3D12GraphicsCommandList4, 0x8754318e, 0xd3a9, 0x4541, 0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74);
WINADAPTER_IID(ID3D12ShaderCacheSession, 0x28e2495d, 0x0f64, 0x4ae4, 0xa6, 0xec, 0x12, 0x92, 0x55, 0xdc, 0x49, 0xa8);
WINADAPTER_IID(ID3D12Device9, 0x4c80e962, 0xf032, 0x4f60, 0xbc, 0x9e, 0xeb, 0xc2, 0xcf, 0xa1, 0xd8, 0x3c);
WINADAPTER_IID(ID3D12Tools, 0x7071e1f0, 0xe84b, 0x4b33, 0x97, 0x4f, 0x12, 0xfa, 0x49, 0xde, 0x65, 0xc5);
WINADAPTER_IID(ID3D12SDKConfiguration, 0xe9eb5314, 0x33aa, 0x42b2, 0xa7, 0x18, 0xd7, 0x7f, 0x58, 0xb1, 0xf1, 0xc7);
WINADAPTER_IID(ID3D12GraphicsCommandList5, 0x55050859, 0x4024, 0x474c, 0x87, 0xf5, 0x64, 0x72, 0xea, 0xee, 0x44, 0xea);
WINADAPTER_IID(ID3D12GraphicsCommandList6, 0xc3827890, 0xe548, 0x4cfa, 0x96, 0xcf, 0x56, 0x89, 0xa9, 0x37, 0x0f, 0x80);
WINADAPTER_IID(ID3D12Debug, 0x344488b7, 0x6846, 0x474b, 0xb9, 0x89, 0xf0, 0x27, 0x44, 0x82, 0x45, 0xe0);
WINADAPTER_IID(ID3D12Debug1, 0xaffaa4ca, 0x63fe, 0x4d8e, 0xb8, 0xad, 0x15, 0x90, 0x00, 0xaf, 0x43, 0x04);
WINADAPTER_IID(ID3D12Debug2, 0x93a665c4, 0xa3b2, 0x4e5d, 0xb6, 0x92, 0xa2, 0x6a, 0xe1, 0x4e, 0x33, 0x74);
WINADAPTER_IID(ID3D12Debug3, 0x5cf4e58f, 0xf671, 0x4ff1, 0xa5, 0x42, 0x36, 0x86, 0xe3, 0xd1, 0x53, 0xd1);
WINADAPTER_IID(ID3D12Debug4, 0x014b816e, 0x9ec5, 0x4a2f, 0xa8, 0x45, 0xff, 0xbe, 0x44, 0x1c, 0xe1, 0x3a);
WINADAPTER_IID(ID3D12DebugDevice1, 0xa9b71770, 0xd099, 0x4a65, 0xa6, 0x98, 0x3d, 0xee, 0x10, 0x02, 0x0f, 0x88);
WINADAPTER_IID(ID3D12DebugDevice, 0x3febd6dd, 0x4973, 0x4787, 0x81, 0x94, 0xe4, 0x5f, 0x9e, 0x28, 0x92, 0x3e);
WINADAPTER_IID(ID3D12DebugDevice2, 0x60eccbc1, 0x378d, 0x4df1, 0x89, 0x4c, 0xf8, 0xac, 0x5c, 0xe4, 0xd7, 0xdd);
WINADAPTER_IID(ID3D12DebugCommandQueue, 0x09e0bf36, 0x54ac, 0x484f, 0x88, 0x47, 0x4b, 0xae, 0xea, 0xb6, 0x05, 0x3a);
WINADAPTER_IID(ID3D12DebugCommandList1, 0x102ca951, 0x311b, 0x4b01, 0xb1, 0x1f, 0xec, 0xb8, 0x3e, 0x06, 0x1b, 0x37);
WINADAPTER_IID(ID3D12DebugCommandList, 0x09e0bf36, 0x54ac, 0x484f, 0x88, 0x47, 0x4b, 0xae, 0xea, 0xb6, 0x05, 0x3f);
WINADAPTER_IID(ID3D12DebugCommandList2, 0xaeb575cf, 0x4e06, 0x48be, 0xba, 0x3b, 0xc4, 0x50, 0xfc, 0x96, 0x65, 0x2e);
WINADAPTER_IID(ID3D12SharingContract, 0x0adf7d52, 0x929c, 0x4e61, 0xad, 0xdb, 0xff, 0xed, 0x30, 0xde, 0x66, 0xef);
WINADAPTER_IID(ID3D12InfoQueue, 0x0742a90b, 0xc387, 0x483f, 0xb9, 0x46, 0x30, 0xa7, 0xe4, 0xe6, 0x14, 0x58);

// DXCore (DXGI is used when building for Windows)
#ifdef __dxcore_interface_h__
WINADAPTER_IID(IDXCoreAdapterFactory, 0x78ee5945, 0xc36e, 0x4b13, 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06);
WINADAPTER_IID(IDXCoreAdapterList, 0x526c7776, 0x40e9, 0x459b, 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28);
WINADAPTER_IID(IDXCoreAdapter, 0xf0db4c7f, 0xfe5a, 0x42a2, 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e);
#endif

// DirectML
WINADAPTER_IID(IDMLObject, 0xc8263aac, 0x9e0c, 0x4a2d, 0x9b, 0x8e, 0x00, 0x75, 0x21, 0xa3, 0x31, 0x7c);
WINADAPTER_IID(IDMLDevice, 0x6dbd6437, 0x96fd, 0x423f, 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f);
WINADAPTER_IID(IDMLDeviceChild, 0x27e83142, 0x8165, 0x49e3, 0x97, 0x4e, 0x2f, 0xd6, 0x6e, 0x4c, 0xb6, 0x9d);
WINADAPTER_IID(IDMLPageable, 0xb1ab0825, 0x4542, 0x4a4b, 0x86, 0x17, 0x6d, 0xde, 0x6e, 0x8f, 0x62, 0x01);
WINADAPTER_IID(IDMLOperator, 0x26caae7a, 0x3081, 0x4633, 0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d);
WINADAPTER_IID(IDMLDispatchable, 0xdcb821a8, 0x1039, 0x441e, 0x9f, 0x1c, 0xb1, 0x75, 0x9c, 0x2f, 0x3c, 0xec);
WINADAPTER_IID(IDMLCompiledOperator, 0x6b15e56a, 0xbf5c, 0x4902, 0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4);
WINADAPTER_IID(IDMLOperatorInitializer, 0x427c1113, 0x435c, 0x469c, 0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13);
WINADAPTER_IID(IDMLBindingTable, 0x29c687dc, 0xde74, 0x4e3b, 0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc);
WINADAPTER_IID(IDMLCommandRecorder, 0xe6857a76, 0x2e3e, 0x4fdd, 0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53);
WINADAPTER_IID(IDMLDebugDevice, 0x7d6f3ac9, 0x394a, 0x4ac3, 0x92, 0xa7, 0x39, 0x0c, 0xc5, 0x7a, 0x82, 0x17);
WINADAPTER_IID(IDMLDevice1, 0xa0884f9a, 0xd2be, 0x4355, 0xaa, 0x5d, 0x59, 0x01, 0x28, 0x1a, 0xd1, 0xd2);
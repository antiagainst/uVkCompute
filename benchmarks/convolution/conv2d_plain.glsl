#version 450 core
#extension GL_EXT_control_flow_attributes : enable

// A shader that computes 2-D convolution.
// - Output: NHoWoCo format
// - Input: NHiWiCi format
// - Filter: HfWfCiCo format
// - N is 1.
// - No padding.
// - No dilation.

layout(local_size_x = 32, local_size_y = 16, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer   { float data[]; } Input;
layout(set=0, binding=1) buffer FilterBufffer { float data[]; } Filter;
layout(set=0, binding=2) buffer OutputBuffer  { float data[]; } Output;

layout(constant_id = 0) const uint OH = 1; // Output height
layout(constant_id = 1) const uint OW = 1; // Output width
layout(constant_id = 2) const uint OC = 1; // Output channel
layout(constant_id = 3) const uint IH = 1; // Input height
layout(constant_id = 4) const uint IW = 1; // Input width
layout(constant_id = 5) const uint IC = 1; // Input channel
layout(constant_id = 6) const uint FH = 1; // Filter height
layout(constant_id = 7) const uint FW = 1; // Filter width
layout(constant_id = 8) const uint SH = 1; // Height stride
layout(constant_id = 9) const uint SW = 1; // Width stride

uint inputCoordToOffset(uint h, uint w, uint c) {
  return h  * IW * IC + w * IC + c;
}

uint filterCoordToOffset(uint h, uint w, uint ic, uint oc) {
  return h * FW * IC * OC + w  * IC * OC + ic * OC + oc;
}

uint outputCoordToOffset(uint h, uint w, uint c) {
  return h  * OW * OC + w * OC + c;
}

// For output               N x OH x OW x OC,
// each invocation computes 1 x  1 x  1 x OC.
// Launch mapping:          z    y    x
void main() {
  uint oh = gl_GlobalInvocationID.y;
  uint ow = gl_GlobalInvocationID.x;

  if (oh >= OH || ow >= OW) return;

  // Initialize the output elements.
  for (uint oc = 0; oc < OC; ++oc) {
    Output.data[outputCoordToOffset(oh, ow, oc)] = 0.f;
  }

  for (uint fh = 0; fh < FH; ++fh) {
    for (uint fw = 0; fw < FW; ++fw) {
      for (uint ic = 0; ic < IC; ++ic) {
        for (uint oc = 0; oc < OC; ++oc) {
          uint ih = oh * SH + fh;
          uint iw = ow * SW + fw;
          Output.data[outputCoordToOffset(oh, ow, oc)] +=
            Input.data[inputCoordToOffset(ih, iw, ic)] *
            Filter.data[filterCoordToOffset(fh, fw, ic, oc)];
        }
      }
    }
  }
}

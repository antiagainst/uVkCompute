// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <memory>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "uvkc/benchmark/dispatch_void_shader.h"
#include "uvkc/benchmark/main.h"
#include "uvkc/benchmark/status_util.h"
#include "uvkc/benchmark/vulkan_context.h"
#include "uvkc/vulkan/device.h"

static const char kBenchmarkName[] = "dispatch_void_shader";

namespace uvkc {
namespace benchmark {

absl::StatusOr<std::unique_ptr<VulkanContext>> CreateVulkanContext() {
  return CreateDefaultVulkanContext(kBenchmarkName);
}

void RegisterVulkanBenchmarks(VulkanContext *context) {
  for (int di = 0; di < context->devices.size(); ++di) {  // GPU
    const char *gpu_name = context->physical_devices[di].properties.deviceName;
    RegisterDispatchVoidShaderBenchmark(
        gpu_name, context->devices[di].get(),
        &context->void_dispatch_latency_seconds);
  }
}

}  // namespace benchmark
}  // namespace uvkc
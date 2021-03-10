"""Loads OpenCL-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "opencl_headers",
        strip_prefix = "OpenCL-Headers-23710f1b99186065c1768fc3098ba681adc0f253",
        sha256 = "5b68923ef7704bce3df7f4a9b9b8ff9b99ed8c07ec27fcf518dcf30f3da2bbc4",
        urls = [
            "https://mirror.bazel.build/github.com/KhronosGroup/OpenCL-Headers/archive/23710f1b99186065c1768fc3098ba681adc0f253.tar.gz",
            "https://github.com/KhronosGroup/OpenCL-Headers/archive/23710f1b99186065c1768fc3098ba681adc0f253.tar.gz",
        ],
        build_file = "//third_party/opencl_headers:BUILD.bazel",
    )

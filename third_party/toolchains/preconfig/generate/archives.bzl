load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bazel_toolchains_archive():
    http_archive(
        name = "bazel_toolchains",
        sha256 = "93282ee0dd9bc1942fbdf781520495c472ce958d4756e89ed3d5d024e7b71d66",
        strip_prefix = "bazel-toolchains-27f2db256e54e5748ee1cd9485ccd0d5444bf1c6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/27f2db256e54e5748ee1cd9485ccd0d5444bf1c6.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/27f2db256e54e5748ee1cd9485ccd0d5444bf1c6.tar.gz",
        ],
    )

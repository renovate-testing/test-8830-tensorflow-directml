"""Loads pasta python package."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pasta",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pasta/archive/v0.2.0.tar.gz",
            "https://github.com/google/pasta/archive/v0.2.0.tar.gz",
        ],
        strip_prefix = "pasta-0.2.0",
        sha256 = "b9e3bcf5ab79986e245c8a2f3a872d14c610ce66904c4f16818342ce81cf97d2",
        build_file = "//third_party/pasta:BUILD.bazel",
        system_build_file = "//third_party/pasta:BUILD.system",
    )

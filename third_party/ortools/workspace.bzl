"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ortools_archive",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/or-tools/archive/v6.9.1.tar.gz",
            "https://github.com/google/or-tools/archive/v6.9.1.tar.gz",
        ],
        sha256 = "c6fa58741addf9fa59dd990fa933a8137eea7e8e2775b09c430cc7f71dbfd564",
        strip_prefix = "or-tools-6.9.1/src",
        build_file = "//third_party/ortools:BUILD.bazel",
    )

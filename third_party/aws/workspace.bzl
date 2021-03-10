"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

# NOTE: version updates here should also update the major, minor, and patch variables declared in
# the  copts field of the //third_party/aws:aws target

def repo():
    third_party_http_archive(
        name = "aws",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aws/aws-sdk-cpp/archive/1.8.158.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.8.158.tar.gz",
        ],
        sha256 = "7127b09aaf336136e98445a5d775ab657966e544c8f839704b16b35336306b38",
        strip_prefix = "aws-sdk-cpp-1.8.158",
        build_file = "//third_party/aws:BUILD.bazel",
    )

package(default_visibility = ["//im2txt:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//im2txt/evaluation/...",
    ],
)

py_binary(
    name = "evaluate_im2txt",
    srcs = ["evaluate_im2txt.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//im2txt/evaluation/pycocoevalcap:eval",
        "//im2txt/evaluation/pycocotools:coco",
    ],
)
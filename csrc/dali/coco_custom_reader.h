// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#define _GLIBCXX_USE_CXX11_ABI 0

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <istream>
#include <memory>

#include "json11.hpp"

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/pipeline/operators/reader/loader/coco_loader.h"
#include "dali/pipeline/operators/reader/parser/coco_parser.h"

using namespace dali;
using namespace json11;

namespace retinanet {

class COCOCustomReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOCustomReader(const OpSpec& spec)
  : DataReader<CPUBackend, ImageLabelWrapper>(spec) {
    auto shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    auto annotations_filename = spec.GetRepeatedArgument<std::string>("annotations_file");

    DALI_ENFORCE(!skip_cached_images_,
      "COCOCustomReader doesn't support `skip_cached_images` option");

    for (auto &filename : annotations_filename) {
      std::ifstream t(filename);
      std::string annotations_json((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
      std::string error;
      Json annotations = Json::parse(annotations_json, error);

      std::vector<int> categories;
      for (auto &category : annotations["categories"].array_items()) {
        categories.push_back(category["category_id"].int_value());
        std::cout << category["category_id"].int_value() << std::endl;
      }
    
    }

    parser_.reset(new COCOParser(spec, annotations_multimap_));
  }

  void RunImpl(SampleWorkspace* ws, const int i) override {
    parser_->Parse(GetSample(ws->data_idx()), ws);
  }

 protected:
  AnnotationMap annotations_multimap_;

  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);
};

}

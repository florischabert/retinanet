/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

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
    auto ltrb = spec.GetArgument<bool>("ltrb");
    auto ratio = spec.GetArgument<bool>("ratio");
    auto size_threshold = spec.GetArgument<float>("size_threshold");

    DALI_ENFORCE(!skip_cached_images_,
      "COCOCustomReader doesn't support `skip_cached_images` option");

    for (auto &filename : annotations_filename) {
      // Parse JSON annotations
      std::ifstream jsonFile(filename);
      std::string annotationsJson((
        std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
      std::string err;
      Json annotations = Json::parse(annotationsJson, err);

      // Get categories mapping
      std::unordered_map<int, int> categories;
      int counter = 0;
      for (auto &category : annotations["categories"].array_items()) {
        categories[category["id"].int_value()] = counter++;
      }
      
      // Get images sizes
      std::unordered_map<int, std::pair<int, int> > image_id_to_wh;
      for (auto &image : annotations["images"].array_items()) {
        image_id_to_wh.insert(std::make_pair(image["id"].int_value(), 
          std::make_pair(image["width"].int_value(), image["height"].int_value())));
      }

      // Parse annotations
      for (auto &annotation : annotations["annotations"].array_items()) {
        auto image_id = annotation["image_id"].int_value();
        auto category = categories[annotation["category_id"].int_value()];

        std::vector<float> bbox;
        for (auto &val : annotation["bbox"].array_items()) {
          bbox.push_back(val.number_value());
        }

        if (bbox[2] < size_threshold || bbox[3] < size_threshold) {
          continue;
        }

        if (ltrb) {
          bbox[2] += bbox[0];
          bbox[3] += bbox[1];
        }

        if (ratio) {
          const auto& wh = image_id_to_wh[image_id];
          bbox[0] /= static_cast<float>(wh.first);
          bbox[1] /= static_cast<float>(wh.second);
          bbox[2] /= static_cast<float>(wh.first);
          bbox[3] /= static_cast<float>(wh.second);
        }

        annotations_multimap_.insert(std::make_pair(image_id,
            Annotation(bbox[0], bbox[1], bbox[2], bbox[3], category)));
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

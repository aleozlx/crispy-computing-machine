// 2018, Patrick Wieschollek <mail@patwie.com>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <iostream>
#include <string>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn,
                             std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph
  status = sess->Create(graph_def.graph_def());
  if (status != tensorflow::Status::OK()) return status;

  // restore model from checkpoint, iff checkpoint is given
  if (checkpoint_fn != "") {
    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING,
                                            tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpoint_fn;

    tensor_dict feed_dict = {
        {graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
    status = sess->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()},
                       nullptr);
    if (status != tensorflow::Status::OK()) return status;
  } else {
    // virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
    //                  const std::vector<string>& output_tensor_names,
    //                  const std::vector<string>& target_node_names,
    //                  std::vector<Tensor>* outputs) = 0;
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  }

  return tensorflow::Status::OK();
}

int main(int argc, char const *argv[]) {
  const std::string graph_fn = "../exported/my_model.meta";
  const std::string checkpoint_fn = "../exported/my_model";

  // prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  TF_CHECK_OK(LoadModel(sess, graph_fn, checkpoint_fn));

  // prepare inputs
  tensorflow::TensorShape data_shape({1, 2});
  tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);

  // same as in python file
  auto data_ = data.flat<float>().data();
  data_[0] = 42;
  data_[1] = 43;

  tensor_dict feed_dict = {
      {"input_plhdr", data},
  };

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(
      sess->Run(feed_dict, {"sequential/Output_1/Softmax:0"}, {}, &outputs));

  std::cout << "input           " << data.DebugString() << std::endl;
  std::cout << "output          " << outputs[0].DebugString() << std::endl;

  return 0;
}

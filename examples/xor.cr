require "../src/micrograd"

# :nodoc:
module XorNN
  alias NNFloat = Float32
  alias NNValue = MicroGrad::Value(NNFloat)
  alias Neuron = MicroGrad::Neuron::Sigmoid(NNFloat)
  alias NN = MicroGrad::MLP(NNFloat, Neuron)

  training_inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
  training_outputs = [[0.0], [1.0], [1.0], [0.0]]
  training_indices = (0...training_inputs.size).to_a

  num_inputs = training_inputs.first.size
  num_hidden = 2
  num_outputs = training_outputs.first.size

  model = NN.new(num_inputs, [num_hidden, num_outputs])

  learning_rate = -0.1
  epochs = 10000

  epochs.times do
    # shuffle indices so we feed training input in random order
    # and cycle through the input
    training_indices.shuffle.each do |x|
      tr_input = training_inputs[x]
      tr_output = training_outputs[x]

      # forward pass
      pred = model.activate!(tr_input)
      loss = NNValue[0]
      tr_output.zip(pred) do |ycorrect, yout|
        loss += (yout - ycorrect)**2
      end

      # backward pass
      model.zero_grad!
      loss.backward

      # update
      model.parameters.each do |p|
        p.data += learning_rate * p.grad
      end
    end
  end

  training_indices.shuffle.each_with_index do |x, i|
    tr_input = training_inputs[x]
    tr_output = training_outputs[x]

    pred = model.activate!(tr_input)
    puts "#{tr_input} => #{pred.first} <== #{tr_output}"

    if i == training_indices.size - 1
      # Generates a graphviz dot file
      gio = File.new("xor_dag.dot", "w")
      pred.first.draw_dot(gio)
      gio.close
    end
  end
end

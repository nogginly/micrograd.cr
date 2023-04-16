require "../src/micrograd"

# :nodoc:
module LectureNN
  alias NNFloat = Float32
  alias NNValue = MicroGrad::Value(NNFloat)
  alias Neuron = MicroGrad::Neuron::TanH(NNFloat)
  alias NN = MicroGrad::MLP(NNFloat, Neuron)

  def self.calculate_loss(ytarget, ypred)
    loss = NNValue[0]
    ytarget.zip(ypred) do |ycorrect, yout|
      loss += (yout.first - ycorrect)**2
    end
    loss
  end

  x = [2.0, 3.0, -1.0]
  n = NN.new(3, [4, 4, 1])
  ypred = n.activate!(x)

  xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]
  ys = [1.0, -1.0, -1.0, 1.0] # desired targets

  epochs = 100
  learning_rate = -0.1

  epochs.times do |k|
    # forward pass
    ypred = xs.map do |x|
      n.activate!(x)
    end

    loss = calculate_loss(ys, ypred)

    # backward pass
    n.zero_grad!
    loss.backward

    # update
    n.parameters.each do |p|
      p.data += learning_rate * p.grad
    end

    # puts("#{k}. #{loss.data}")
  end

  # predict
  ypred = xs.map do |x|
    n.activate!(x)
  end
  loss = calculate_loss(ys, ypred)
  puts("loss #{loss.data}")

  ypred.flatten.each do |yp|
    puts yp
  end

  # Generates a graphviz dot file
  gio = File.new("lecture_dag.dot", "w")
  loss.draw_dot(gio)
  gio.close
end

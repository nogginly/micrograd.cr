require "./engine"

module MicroGrad
  # Shared functionality for neuron, layer and MLP.
  module Common(T)
    # Zero the gradient of all the parameters
    def zero_grad!
      parameters.each do |p|
        p.grad = 0
      end
    end

    # Memo when retrieving parameters; internal use
    @parameters : Array(Value(T))?

    # Retrieve references to all the parameters as an array
    abstract def parameters : Array(Value(T))

    # Activate the neuron, layer or network with the given *inputs*
    abstract def activate!(inputs)

    # Prints a concise string representation, typically intended for users, to *io*
    abstract def to_s(io)
  end

  # Base neuron, always linear; use the sub-classes to enable non-linear activation using the desired function.
  class Neuron(T)
    include Common(T)
    getter w : Array(Value(T))
    getter b : Value(T)

    # Create a linear neuron with *num_inputs*
    def initialize(num_inputs)
      raise ArgumentError.new unless num_inputs.positive?

      @w = Array(Value(T)).new(num_inputs) do
        Value(T)[Random.rand * 2.0 - 1.0] # random uniform range -1.0 .. 1.0
      end
      @b = Value(T)[0]
    end

    # :inherit:
    def parameters : Array(Value(T))
      @parameters ||= w + [b]
    end

    # :inherit:
    def activate!(inputs) : Value(T)
      act = b
      w.zip(inputs) do |xi, wi|
        act += wi * xi
      end
      act
    end

    # Print name to *io*
    protected def name(io)
      io << "Neuron/Linear"
    end

    # :inherit:
    def to_s(io)
      name(io)
      io << '(' << w.size << ')'
    end
  end

  # Non-linear neuron that uses `tanh`
  class Neuron::TanH(T) < Neuron(T)
    # :inherit:
    protected def name(io)
      io << "Neuron/TanH"
    end

    # :inherit:
    def activate!(inputs) : Value(T)
      super.tanh
    end
  end

  # Non-linear neuron that uses `ReLU`
  class Neuron::ReLU(T) < Neuron(T)
    # :inherit:
    protected def name(io)
      io << "Neuron/ReLu"
    end

    # :inherit:
    def activate!(inputs) : Value(T)
      super.relu
    end
  end

  # Non-linear neuron that uses `sigmoid`
  class Neuron::Sigmoid(T) < Neuron(T)
    # :inherit:
    protected def name(io)
      io << "Neuron/Sigmoid"
    end

    # :inherit:
    def activate!(inputs) : Value(T)
      super.sigmoid
    end
  end

  # A single layer in the neural network, specified using the `T` data type and `N` neuron
  class Layer(T, N)
    include Common(T)
    getter neurons : Array(N)

    # Create a layer with *num_inputs* and *num_outputs*
    def initialize(num_inputs, num_outputs)
      @neurons = Array(N).new(num_outputs) do
        N.new(num_inputs)
      end
    end

    # :inherit:
    def parameters : Array(Value(T))
      @parameters ||= neurons.flat_map do |n|
        n.parameters
      end
    end

    # :inherit:
    def activate!(inputs)
      neurons.map &.activate!(inputs)
    end

    # :inherit:
    def to_s(io)
      io << "Layer("
      neurons.each do |n|
        n.to_s(io)
        io << ", "
      end
      io << ')'
    end
  end

  class MLP(T, N)
    include Common(T)
    getter layers : Array(Layer(T, N))

    # Create a multi-layer perceptron with *num_inputs* and multiple layers with sizes in
    # *num_output_list*.
    def initialize(num_inputs, num_output_list)
      nin = num_inputs
      @layers = num_output_list.map do |nout|
        Layer(T, N).new(nin, nout)
      end
    end

    # :inherit:
    def parameters : Array(Value(T))
      @parameters ||= layers.flat_map do |lyr|
        lyr.parameters
      end
    end

    # :inherit:
    def activate!(inputs)
      x = inputs.map { |inp| Value(T)[inp] }
      layers.each do |lyr|
        x = lyr.activate!(x)
      end
      x
    end

    # :inherit:
    def to_s(io)
      io << "MLP("
      layers.each do |lyr|
        lyr.to_s(io)
        io << ", "
      end
      io << ')'
    end
  end
end

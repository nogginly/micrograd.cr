abstract struct Number
  # Subtract a `MicroGrad::Value` from this scalar and return the result as a `MicroGrad::Value` chained with the inputs.
  def -(other : MicroGrad::Value)
    #    self - other
    # => self + (-other)
    self + (-other)
  end

  # Divide this scalar by a `MicroGrad::Value` and return the result as a `MicroGrad::Value` chained with the inputs.
  def /(other : MicroGrad::Value)
    #    self / other
    # => self * (1 / other)
    # => self * (other ** -1)
    self * (other ** -1)
  end

  # Multiply this scalar by a `MicroGrad::Value` and return the result as a `MicroGrad::Value` chained with the inputs.
  def *(other : MicroGrad::Value)
    #    self * other
    # => other * self
    other * self
  end

  # Add this scalar to a `MicroGrad::Value` and return the result as a `MicroGrad::Value` chained with the inputs.
  def +(other : MicroGrad::Value)
    #    self + other
    # => other + self
    other + self
  end
end

module MicroGrad
  # Stores a single scalar value and its gradient
  class Value(T)
    getter data : T
    getter grad : T = T.new(0)

    @_prev : Tuple(Value(T), Value(T)?)?
    @_op : String?
    @_backward : Proc(Nil)?

    # Initialize a `Value(T)` with a `Number`; useful for casting different scalars
    # into a `Value`
    def self.[](num : Number)
      self.new(T.new(num))
    end

    # Create a `Value`. For internal use.
    protected def initialize(@data, @_prev = nil, @_op = nil); end

    # Assign a gradient
    def grad=(value : Number)
      @grad = T.new(value)
    end

    # Assign a data value
    def data=(value : Number)
      @data = T.new(value)
    end

    # Assign the backward (derivative) closure; for internal use
    protected def _backward(&proc : ->)
      @_backward = proc
    end

    # Add this `Value` to another and return the result as a `Value` chained with the inputs.
    def +(other : Value)
      result = Value.new(data + other.data, {self, other}, "+")
      result._backward do
        self.grad += result.grad
        other.grad += result.grad
      end
      result
    end

    # Multiply this `Value` with another and return the result as a `Value` chained with the inputs.
    def *(other : Value)
      result = Value.new(data * other.data, {self, other}, "*")
      result._backward do
        self.grad += other.data * result.grad
        other.grad += self.data * result.grad
      end
      result
    end

    # Subtract another `Value` from this one and return the result as a `Value` chained with the inputs.
    def -(other : Value)
      self + (-other)
    end

    # Divide this `Value` by another and return the result as a `Value` chained with the inputs.
    def /(other : Value)
      self * (other ** -1)
    end

    # Negate this `Value` and return the result as a `Value` chained with the inputs.
    def -
      self * T.new(-1)
    end

    # Add this `Value` to any `Number` (internally wrapping that in a `Value`) and return the result as a `Value` chained with the inputs.
    def +(other : Number)
      self + self.class[other]
    end

    # Multiple this `Value` with any `Number` (internally wrapping that in a `Value`) and return the result as a `Value` chained with the inputs.
    def *(other : Number)
      self * self.class[other]
    end

    # Divide this `Value` by any `Number` (internally wrapping that in a `Value`) and return the result as a `Value` chained with the inputs.
    def /(other : Number)
      self / self.class[other]
    end

    # Subtract any `Number` from this `Value` (internally wrapping that in a `Value`) and return the result as a `Value` chained with the inputs.
    def -(other : Number)
      self + (-other)
    end

    # Calculate this value's data to the power of the given scalar and return the result as a `Value` chained with the inputs.
    def **(other : Number)
      result = Value.new(data ** other, {self, nil}, "** #{other}")
      result._backward do
        self.grad += (other * data**(other - 1)) * result.grad
      end
      result
    end

    # ReLU
    def relu
      result = Value.new(data < T.new(0) ? T.new(0) : data, {self, nil}, "ReLU")

      result._backward do
        self.grad += result.data > 0 ? result.grad : T.new(0)
      end
      result
    end

    # *tanh*
    def tanh
      t = Math.tanh(data)
      result = Value.new(t, {self, nil}, "tanh")

      result._backward do
        self.grad += (1 - (t ** 2)) * result.grad
      end
      result
    end

    # Power of *e*
    def exp
      e = Math.exp(data)
      result = Value.new(e, {self, nil}, "tanh")

      result._backward do
        self.grad += e * result.grad
      end
      result
    end

    # Natural log
    def log
      x = data
      result = Value.new(Math.log(x), {self, nil}, "log")

      result._backward do
        self.grad += (T.new(1) / x) * result.grad
      end
      result
    end

    # Sigmoid
    def sigmoid
      s = T.new(1) / (T.new(1) + Math.exp(-data))
      result = Value.new(s, {self, nil}, "sigmoid")

      result._backward do
        self.grad += (s * (T.new(1) - s)) * result.grad
      end
      result
    end

    # Return a string describing the data and gradient of this `Value`
    def to_s(io)
      io << "Value(#{data}, g=#{grad})"
    end

    # Calculate the gradients backwards from this `Value`
    def backward
      topo = build_topo
      # go bakward one variable at a time and apply the chain rule to get its gradient
      self.grad = T.new(1)
      topo.reverse_each do |v|
        v.@_backward.try &.call
      end
    end

    # Return array of topologically ordered values; internal use only
    protected def build_topo(topo = Array(self).new, visited = Set(self).new)
      unless visited.includes?(self)
        visited << self
        @_prev.try &.each do |child|
          child.try &.build_topo(topo, visited)
        end
        topo << self
      end
      topo
    end

    # Generate a Graphviz `dot` digraph diagram to the *io*; the output can be converted to an image
    # using the `dot` tool.
    def draw_dot(io, graph_name = "X")
      io.puts "digraph #{graph_name} {"
      io.puts "dpi=192; rankdir=LR; ranksep=0.25"
      io.puts "node [shape=record fontsize=8 margin=0.05 height=0 width=0]"
      io.puts "edge [fontsize=9 arrowsize=0.5]"
      to_dot(io)
      io.puts "}"
    end

    # Recursively generate the `dot` diagram for the `Value` and it's
    # source (previous) graph; for internal use only
    protected def to_dot(io, visited = Set(self).new)
      return if visited.includes?(self)
      visited << self
      io << "\"#{hash}\" [label=\"{{data | #{data}}|{grad | #{grad}}}\"]\n"
      if prev = @_prev
        if op = @_op
          op_id = "#{op}#{hash}".hash
          io << "\"#{op_id}\" [shape=oval, label=\"#{op}\"]\n"
          (first = prev.first).to_dot(io, visited)
          if second = prev.last
            second.to_dot(io, visited)
            io << "\"#{second.hash}\" -> \"#{op_id}\"\n"
          end
          io << "\"#{first.hash}\" -> \"#{op_id}\"\n"
          io << "\"#{op_id}\" -> \"#{hash}\"\n"
        end
      end
    end
  end
end

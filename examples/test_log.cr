require "../src/micrograd"

alias NNFloat = Float32
alias NNValue = MicroGrad::Value(NNFloat)

x = [0.1, 0.2, 0.3].map { |v| NNValue[v] }
puts "x = #{x.map(&.data)}"

l = x.map { |v| v.log }
puts "log(x) = #{l.map(&.data)}"

s = l.sum(NNValue[0])
puts "sum(log(x)) = #{s.data}"

s.backward
puts "x.grad = #{x.map(&.grad)}"

# Generates a graphviz dot file
gio = File.new("test_log.dot", "w")
s.draw_dot(gio)
gio.close

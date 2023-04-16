require "../src/micrograd"

alias NNFloat = Float32
alias NNValue = MicroGrad::Value(NNFloat)

a = NNValue[-4]
b = NNValue[2]
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu
d += 3 * d + (b - a).relu
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

puts "g: #{g} (expect data = 24.7041)" # prints 24.7041, the outcome of this forward pass
g.backward
puts "a: #{a} (expect grad = 138.8338)" # prints 138.8338, i.e. the numerical value of dg/da
puts "b: #{b} (expect grad = 645.5773)" # prints 645.5773, i.e. the numerical value of dg/db

# Generates a graphviz dot file
gio = File.new("compat_dag.dot", "w")
g.draw_dot(gio)
gio.close

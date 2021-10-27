from py3do.path import Path

p = Path()
p.to(0,1).to(1,1).to(1,0).fd(0.5).lt(90).fd(0.5).rt(90)


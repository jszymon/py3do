from py3do.path import Path

p = Path(closed=True)
p.to(0,1).to(1,1).to(1,0).fd(0.5).lt(90).fd(0.5).rt(90)
#p.plot()

p = Path(closed=True, round_r=0.05)
p.to(0,1).to(1,1).round(0.3).to(1,0).fd(0.5).round(None).lt(90).fd(0.5).rt(90)
p.plot()

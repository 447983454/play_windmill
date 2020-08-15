import taichi as ti
import numpy as np
import math
ti.init(arch=ti.cpu) # Try to run on GPU
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
x = ti.Vector(2, dt=ti.f32, shape=n_particles) # position
v = ti.Vector(2, dt=ti.f32, shape=n_particles) # velocity
C = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles) # affine velocity field
F = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles) # deformation gradient
material = ti.var(dt=ti.i32, shape=n_particles) # material id
Jp = ti.var(dt=ti.f32, shape=n_particles) # plastic deformation
grid_v = ti.Vector(2, dt=ti.f32, shape=(n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid)) # grid node mass

# windmill
sail_num = 2
pi = math.pi
radius = 0.2
thick = 0.015
center = ti.Vector(2, dt=ti.f32, shape=())
angle = ti.var(dt=ti.f32, shape=())
omega = ti.var(dt=ti.f32, shape=())
pos = ti.Vector(2, dt=ti.f32, shape=(sail_num*2))   # sail position, using to draw lines in gui
transform = ti.Matrix(3, 3, dt=ti.f32, shape=())
transforms = ti.Matrix(3, 3, dt=ti.f32, shape=(sail_num))
windmill_n_particles = 2000
windmill_x = ti.Vector(2, dt=ti.f32, shape=windmill_n_particles) # position
# windmill_C = ti.Matrix.zero(ti.f32, 2, 2)
# windmill_F = ti.Matrix([1., 0.],[0., 1.])
torque = ti.var(dt=ti.f32, shape=())
I = 1e-2
grid_assit = ti.Vector(2, dt=ti.f32, shape=(n_grid, n_grid)) # grid node momentum/velocity



@ti.kernel
def substep():
  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_m[i, j] = 0
    grid_assit[i, j] = [0, 0]
  for p in x: # Particle state update and scatter to grid (P2G)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p] # deformation gradient update
    h = ti.exp(10 * (1.0 - Jp[p])) # Hardening coefficient: snow gets harder when compressed
    if material[p] == 1: # jelly, make it softer
      h = 1.0
    mu, la = mu_0 * h, lambda_0 * h
    if material[p] == 0: # liquid
      mu = 0.0
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(2)):
      new_sig = sig[d, d]
      if material[p] == 2:  # Snow
        new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig
    if material[p] == 0:  # Reset deformation gradient to avoid numerical instability
      F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
    elif material[p] == 2:
      F[p] = U @ sig @ V.T() # Reconstruct elastic deformation gradient after plasticity
    stress = 2 * mu * (F[p] - U @ V.T()) @ F[p].T() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      grid_assit[base + offset] += weight * (affine @ dpos)
      grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
      grid_m[base + offset] += weight * p_mass

  for p in windmill_x:
    windmill_C = ti.Matrix.zero(ti.f32, 2, 2)
    windmill_F = ti.Matrix.identity(ti.f32, 2)  #ti.Matrix([1., 0.],[0., 1.])

    base = (windmill_x[p] * inv_dx - 0.5).cast(int)
    fx = windmill_x[p] * inv_dx - base.cast(float)
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    # F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p] # deformation gradient update
    h = 1.0 # Hardening coefficient: snow gets harder when compressed
    mu, la = mu_0 * h, lambda_0 * h
    U, sig, V = ti.svd(windmill_F)
    J = 1.0 
    stress = 2 * mu * (windmill_F - U @ V.T()) @ windmill_F.T() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * windmill_C
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      wr = ti.Vector([center[None][1]-windmill_x[p][1], windmill_x[p][0]-center[None][0]])
      # wr = wr/wr.norm()
      wr = wr*omega[None]/180.*pi
      grid_v[base + offset] += weight * (p_mass * wr + affine @ dpos)
      grid_m[base + offset] += weight * p_mass

  # torque[None] = 0.
  # for p in windmill_x:
  #   base = (windmill_x[p] * inv_dx - 0.5).cast(int)
  #   fx = windmill_x[p] * inv_dx - base.cast(float)
  #   # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
  #   w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
  #   for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
  #     offset = ti.Vector([i, j])
  #     dpos = (offset.cast(float) - fx) * dx
  #     weight = w[i][0] * w[j][1]
  #     torque[None] += weight * grid_assit[base + offset].dot(windmill_x[p]-center[None])
  # print(torque[None])

  for i, j in grid_m:
    if grid_m[i, j] > 0: # No need for epsilon here
      grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
      grid_v[i, j][1] -= dt * 50 # gravity
      if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
      if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
      if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
  for p in x: # grid to particle (G2P)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(ti.f32, 2)
    new_C = ti.Matrix.zero(ti.f32, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection

  torque[None] = 0.
  for p in windmill_x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(ti.f32, 2)
    for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
    torque[None] += new_v.dot(windmill_x[p]-center[None])
  print(torque[None])
    

@ti.func
def get_transform_matrix(a):
  A = ti.Matrix([[1., 0., -center[None][0]], [0., 1., -center[None][1]], [0., 0., 1.]])
  theta = a / 180. * pi
  A2 = ti.Matrix([[ti.cos(theta), -ti.sin(theta), 0.], 
                  [ti.sin(theta), ti.cos(theta), 0.], 
                  [0., 0., 1.]])
  A2 = A2@A
  A = ti.Matrix([[1., 0., center[None][0]], [0., 1., center[None][1]], [0., 0., 1.]])
  transform[None] = A@A2

group_size = n_particles # // 3

@ti.kernel
def initialize():
  for i in range(n_particles):
    x[i] = [ti.random() * 0.5 + 0.25 + 0.10 * (i // group_size), ti.random() * 0.5 + 0.3 + 0.32 * (i // group_size)]
    material[i] = i // group_size # 0: fluid 1: jelly 2: snow
    v[i] = ti.Matrix([0, 0])
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1

  # initialize windmill
  center[None] = [0.3, 0.25]
  angle[None] = 0.
  omega[None] = 1000.
  step = 360. / sail_num / 2.
  for i in range(sail_num):
    get_transform_matrix(i*step)
    transforms[i] = transform[None]
    print(transform[None])
  for i in range(sail_num):
    print(transforms[i])  
  group_num = windmill_n_particles // sail_num
  for i in range(windmill_n_particles):
    p = ti.Vector([ti.random() * (2.*radius) + center[None][0] - radius, 
                     ti.random() * (2.*thick) + center[None][1] - thick, 
                     1.])
    p = transforms[i//group_num]@p
    windmill_x[i] = [p[0], p[1]]
  for j in range(sail_num*2):
    get_transform_matrix(j*step)
    p = transform[None]@ti.Vector([center[None][0]+radius, center[None][1], 1]) 
    pos[j] = [p[0], p[1]]
    # windmill particles


@ti.kernel
def solve_windmill():
  omega[None] = torque[None] / I  # Constant Velocity ?
  angle[None] += omega[None]*dt
  

  for p in windmill_x:
    # vertex
    # p = transform[None]@ti.Vector([pos[i][0], pos[i][1], 1.])
    # pos[i] = [p[0], p[1]]

    wr = ti.Vector([center[None][1]-windmill_x[p][1], windmill_x[p][0]-center[None][0]])
    # wr = wr/wr.norm()
    wr = wr*omega[None]/180.*pi
    windmill_x[p] += wr*dt

initialize()
gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
for frame in range(500):
  for s in range(int(2e-3 // dt)):
    substep()
    solve_windmill()

  colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
  gui.circles(x.to_numpy(), radius=1.5, color=colors[material.to_numpy()])

  # draw windmill
  pos_n = pos.to_numpy()
  # for i in range(sail_num):
    # gui.line(pos_n[i], pos_n[i+sail_num], radius=5)
  gui.circles(windmill_x.to_numpy(), radius=1.5)
  # gui.show() # Change to gui.show(f'images/{frame:06d}.png') to write images to disk
  gui.show(f'images2/{frame:06d}.png')
import numpy as np
import torch
import torch.autograd as autograd

torch.manual_seed(42)
np.random.seed(42)

x_np = np.random.uniform(-5, 7, [30, 5]).astype(np.float32)
x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
out = torch.nn.functional.elu(x, alpha=1.0)
dout = torch.ones_like(out)
(dx,) = autograd.grad(out, x, grad_outputs=dout)


print(f"x:\n{x_np[0]}")
with open("x.bin", "wb") as f:
    x_np.tofile(f)

out_np = out.detach().numpy()
print(f"out:\n{out_np[0]}")
with open("out.bin", "wb") as f:
    out_np.tofile(f)

dout_np = dout.detach().numpy()
print(f"dout:\n{dout_np[0]}")
with open("dout.bin", "wb") as f:
    dout_np.tofile(f)

dx_np = dx.detach().numpy()
print(f"dx:\n{dx_np[0]}")
with open("dx.bin", "wb") as f:
    dx_np.tofile(f)

print("All steps completed and files saved.")

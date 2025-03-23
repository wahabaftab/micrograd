import torch


def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z + z * x
    j = (z * z).relu()
    y = j + q + q * x
    y.backprop(Value(h),y.grad)
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z + z * x
    j = (z * z).relu()
    y = j + q + q * x
    y.backward()
    xpt, ypt = x, y
    print(f"My Output:   ymg.data = {ymg.data:.6f} | PyTorch Output: ypt.data = {ypt.data.item():.6f}")
    print(f"My Gradient: xmg.grad = {xmg.grad:.6f} | PyTorch Gradient: xpt.grad = {xpt.grad.item():.6f}")


    # forward pass went well
    assert round(ymg.data,2) == round(ypt.data.item(),2)
    # backward pass went well
    assert round(xmg.grad,2) == round(xpt.grad.item(),2)

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop(Value(h),g.grad)
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    print(f"My Output: {gmg.data:.6f} | PyTorch Output: {gpt.data.item():.6f}")
    # forward pass went well
    assert abs(round(gmg.data,2) - round(gpt.data.item(),2)) < tol
    # backward pass went well
    print(f"My Gradients     : amg.grad = {amg.grad:.6f}, bmg.grad = {bmg.grad:.6f}")
    print(f"PyTorch Gradients: apt.grad = {apt.grad.item():.6f}, bpt.grad = {bpt.grad.item():.6f}")

    assert abs(round(amg.grad,2) - round(apt.grad.item(),2)) < tol
    assert abs(round(bmg.grad,2) - round(bpt.grad.item(),2)) < tol

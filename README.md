# Perlin Art

Explore Perlin noise with simple art! Visualize how lines following perlin noise appear much more fluid
and can create interesting patterns

![Example 1](./examples/example1.gif)

## Examples
```python
pshapes = grid(n=3, margin=200, shape="circle")
```
![Example 2](./examples/example2.gif)

```python
pshapes = grid(rows=3, columns=5, margin=250, xbounds=150, ybounds=50)
pshapes = [*pshapes, *grid(rows=5, columns=3, margin=250, xbounds=50, ybounds=150)]
```
![Example 3](./examples/example3.gif)

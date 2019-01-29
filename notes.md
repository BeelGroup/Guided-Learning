### OpenAI/Retro quirks
Game data files such as data.json and scenario.json are accessed from:
```
/Users/Keith/anaconda/envs/python3/lib/python3.6/site-packages/retro/data/stable/SuperMarioBros-Nes
```

### Actions:

SUPER MARIO WORLD actions:
* [jump (B), run (Y), (Select), (Start), (Up), (Down), (Left), (Right), power_jump (A), run (X), (L), (R)]

SUPER MARIO BROS actions:
* [run (B), (Select), (Start), (Up), (Down), (Left), (Right), jump (A)]

Creating an action array:
```
ac = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
action = np.array(ac).astype(np.uint8)
```

---

### Observations:
using ob (observation) as the inputs
```
ob.shape=(224, 240, 3) # 3 arrays of R, G & B values for each pixel
```

---

### Matplotlib:
Simple image display:
```
plt.imshow(ob, interpolation='nearest') + , cmap="gray" from greyscale images
plt.show()
```

### Segmentation
As a temp solution I am dividing up the pixels into 16x16 squares that are aligned (more or less) with the tiles of the game.
When I was attempting more general segmentation this performed the best:
```
plt.imshow(felzenszwalb(frame, sigma=1.5), interpolation='nearest', cmap="gray")
```
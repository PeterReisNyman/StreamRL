# VoxelCraft Coordinate Guide 🧭

## How to Find the AI Agents

### Your Spawn Location
When you open VoxelCraft, you spawn at:
- **Position: (0, 120, 0)**
- This is your reference point!

### Agent Locations

```
         North (Z-)
         Charlie 🟦
         (0, 120, -5)
              |
              |
West --------YOU-------- East
(X-)      (0,120,0)      (X+)
         /        \
        /          \
   Bob 🟢          Alice 🔴
(-5,120,5)       (5,120,5)
Southwest      Southeast
              |
         South (Z+)
```

### Where to Look

1. **Alice's STONE Tower** 🔴
   - Location: (5, 120, 5)
   - Direction: Southeast
   - How to find: Look right and forward

2. **Bob's WOOD Tower** 🟢
   - Location: (-5, 120, 5)
   - Direction: Southwest
   - How to find: Look left and forward

3. **Charlie's GRASS Tower** 🔵
   - Location: (0, 120, -5)
   - Direction: North
   - How to find: Look straight back

### VoxelCraft Controls

- **WASD**: Move
- **Mouse**: Look around
- **Space**: Jump/Fly up
- **Shift**: Fly down
- **G**: Toggle free camera (great for viewing from above!)

### Tips for Viewing

1. **Press G** to enter free camera mode
2. **Fly up** to get a bird's eye view
3. **Look down** to see all three towers at once
4. Towers build **upward** (Y+) from ground level

### Coordinate System

- **X-axis**: West (-) to East (+)
- **Y-axis**: Down (-) to Up (+) [Height]
- **Z-axis**: North (-) to South (+)

### Debugging

If you can't see towers:
1. Check the console output - it shows block placement
2. Use free camera (G) and fly up
3. Look for ✨ messages showing block placement coordinates
4. Wait a few turns - agents take time to build

### Example Output

When agents are working, you'll see:
```
[Alice] ✓ Executed MOVE_UP | Now at (5.0, 122.0, 5.0)
✨ Placed STONE at (5, 122, 5) ✨
[Bob] ✓ Executed PLACE_BLOCK | Now at (-5.0, 120.0, 5.0)
✨ Placed WOOD at (-5, 120, 5) ✨
```

This tells you EXACTLY where blocks are being placed!

---
task: "Pick up the ball and place it in the cup."
---

# Ball into a cup (SO101, RGB only)

## Goal and setup

Grasp a light ball (for example a 40 mm table-tennis ball) from the table and release it
inside an upright mug, then verify it rests inside. Read [SO101](../../../docs/so101.md)
for the session, interlocks, tools and frame conventions. Positions below are
URDF-model millimetres in `so101_base`; pitch is degrees below horizontal.

Find the ball numerically first, and re-sample its HSV whenever the light changes. A pale
mint ball first segmented with `so101.detect_color([64, 35, 60], [92, 255, 255],
min_fill=0.6)` when the table's saturation was below 20. Under cooler light the table
read H≈86, S≈35, and only `[60, 80, 80]–[95, 255, 255]` isolated the ball.

## Grasp: slide along the fingers at about 38° pitch

1. From rest, `so101.go_to(xyz_mm=..., pitch=38)` to a staging pose at about z = 50 mm,
   30–60 mm inward of the ball (no closer than radius 220 mm at that height).
2. `set_gripper(0.55)`. The ball should appear between the jaw tips and ahead of them in
   the wrist image, and ahead of the open jaws in the external view. It drifts right,
   toward the moving jaw, as the jaws approach; pan +3–5° (left in the image) to keep
   it near x ≈ 430 at close range.
3. Choose a slide line along the fingers, `f = (cos p · heading, −sin p)`, that ends
   with the tool at z ≈ 0 mm and the ball about 18 mm inside the tips. At this pitch
   the jaw plane is mostly horizontal, so its height comes from the tool z. If the
   external view shows the tips at the ball's near edge, the remaining slide is the
   ball radius plus 18 mm.
4. `so101.close_until_contact(0.1)`. A 40 mm ball stalled at opening 0.20 (load 320–500) in both runs.
   The goal is then eased to about 0.18.
5. Verify the grasp. Lift 40 mm with `so101.move_tool([0, 0, 40])`. The ball's
   wrist-image position and radius should stay unchanged, and the external view should
   show the tissue empty.

Two approaches failed repeatedly, both in ways that look plausible from the wrist
camera:

- Descending vertically at about 38° moves the ball toward the palm by about 0.6 mm per
  mm of descent. The jaws then close above its equator and squeeze it out underneath.
  The tool at z = +6 mm failed this way; −1 mm succeeded.
- At pitch about 75° the wrist camera sits radially outward of the jaw plane and looks
  through it. A ball inward of the plane still appears between the jaws, and the moving
  jaw sweeps in front of it. Five closes missed like this. If the closed jaw occludes
  the ball, the ball is out of the jaw plane. Changing the lateral position will not
  fix it; return to the sliding approach.

## Place and release

1. Lift to tool z ≈ 140 mm; a mug rim is about 90 mm above the table.
2. Estimate the mug position from the external view. Fit a pixel-to-robot map from
   known ball positions (where it was grasped, where it was released) rather than
   eyeballing; keep table-height and rim-height points apart, since mixing them
   overstated a correction by about 50%. At about 38° pitch the ball centre
   is about 14 mm behind the tool horizontally. Move the ball over the mug at that height.
3. Lower the ball to about rim height (tool z ≈ 100 mm). In a zoomed external crop,
   compare the ball with the rim centre; at equal heights parallax cancels. Lift 15 mm
   before any lateral correction, so the ball clears the rim. When aligned, the wrist
   view shows the rim encircling the ball.
4. Release with the ball's bottom near the rim, so a bounce stays inside the mug: open
   to about 0.48 in steps, then lift away.
5. Verify the ball inside the mug in both views; at some camera angles the mug wall
   hides it. Return with `so101.go_to(joints_deg=[-5, -106.3, 66.7, 97.2, -84.8])`;
   `shoulder_lift` must stay above its limit of about −107°.

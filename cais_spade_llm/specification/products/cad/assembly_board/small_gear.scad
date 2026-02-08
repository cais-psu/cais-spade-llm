// CAD model: small_gear
// Single spur gear example using BOSL2

include <BOSL2/std.scad>;
include <BOSL2/gears.scad>;

$fn = 81;  // Smooth circles

// === GEAR PARAMETERS ===
mod              = 1;   // Module (tooth size)
teeth            = 60;     // Number of teeth
thickness = 10;
profile_shift    = 0;      // Profile shift (0 = standard)
shaft_diameter = 22;

// === CYLINDER PARAMETERS ===
cylinder_diameter = 22;
cylinder_height = 10; // Cylinder height (mm)

// === HOLE PARAMETERS ===
hole_diameter   = 10.5;           // Diameter of hole (mm)
hole_height = 10;

// === RENDER SINGLE GEAR ===
difference() {
union() {
// Centers the gear at the origin and sits it on the Z=0 plane
    translate([0, 0, thickness/2])
    spur_gear(
        mod           = mod,
        teeth         = teeth,
        profile_shift = profile_shift,
        shorten       = 0,          // no shortening
        thickness     = thickness,
        shaft_diameter    = shaft_diameter,
        gear_spin     = -90         // orient teeth upright
    );
    
    
      translate([0, 0, (thickness+cyl_height)/2])
    cylinder(
      h = cyl_height,
      d = cyl_diam);

}
      translate([0, 0, thickness])
    cylinder(
      h = hole_height,
      d = hole_diam);
}
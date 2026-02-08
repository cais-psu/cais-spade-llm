// CAD model: large_rectangular_pin
// === RECTANGULAR PIN PARAMETERS ===
pin_length = 300;
pin_width = 16;
pin_height = 10;

// Center it on X/Y and sit on Z=0
translate([-pin_length/2, -pin_width/2, 0])
    cube([pin_length, pin_width, pin_height], center = false);
var int: x_11;
var int: x_12;
var int: x_13;
var int: x_14;
var int: x_21;
var int: x_22;
var int: x_23;
var int: x_24;
var int: x_31;
var int: x_32;
var int: x_33;
var int: x_34;

constraint x_11 + x_12 + x_13 + x_14 = 75;
constraint x_21 + x_22 + x_23 + x_24 = 125;
constraint x_31 + x_32 + x_33 + x_34 = 100;

constraint x_11 + x_21 + x_31 = 80;
constraint x_12 + x_22 + x_32 = 65;
constraint x_13 + x_23 + x_33 = 70;
constraint x_14 + x_24 + x_34 = 85;

constraint x_11 >= 0;
constraint x_12 >= 0;
constraint x_13 >= 0;
constraint x_14 >= 0;
constraint x_21 >= 0;
constraint x_22 >= 0;
constraint x_23 >= 0;
constraint x_24 >= 0;
constraint x_31 >= 0;
constraint x_32 >= 0;
constraint x_33 >= 0;
constraint x_34 >= 0;

solve minimize (464 * x_11) + (513 * x_12) + (654 * x_13) + (867 * x_14) + (352 * x_21) + (416 * x_22) + (690 * x_23) + (791 * x_24) + (995 * x_31) + (682 * x_32) + (388 * x_33) + (685 * x_34) 






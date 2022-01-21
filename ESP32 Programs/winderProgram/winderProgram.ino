// Libraries used in this program
#include <Adafruit_LSM6DS33.h>
#include <ESP_FlexyStepper.h>

// IO Pin assignments
const int MOTOR_STEP_PIN = 19;
const int MOTOR_DIRECTION_PIN = 18;
const int MOTOR_ENA_PIN = 12;
const int STOP_PIN = 5;
const int LED_PIN = 13;

// Speed settings
const int MAX_SPEED = 4500;
const int ACCELERATION_IN_STEPS_PER_SECOND = 500;
const int DECELERATION_IN_STEPS_PER_SECOND = 1000;
const int DISTANCE_TO_TRAVEL_IN_STEPS = -10000;

// create the stepper motor object
ESP_FlexyStepper stepper;

// Global constants
const float GRAVITY = 9.81;

// Global variables
float armAngle = 0;
unsigned long lastTime;
int motorSpeed = 50;

// idk, does something with this library
Adafruit_LSM6DS33 lsm6ds33;

// Setup
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** 
void setup(void) {

  // Try to start serial until started
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  // Try to connect to sensor until connected
  if (!lsm6ds33.begin_I2C()) {
    Serial.println("Failed to find LSM6DS33 chip");
    while (1) {
      delay(10);
    }
  }

  // Sensor setup
  lsm6ds33.configInt1(false, false, true); // accelerometer DRDY on INT1
  lsm6ds33.configInt2(false, true, false); // gyro DRDY on INT2

   // connect and configure the stepper motor to its IO PIns
  stepper.connectToPins(MOTOR_STEP_PIN, MOTOR_DIRECTION_PIN);
  stepper.setSpeedInStepsPerSecond(motorSpeed);
  stepper.setAccelerationInStepsPerSecondPerSecond(ACCELERATION_IN_STEPS_PER_SECOND);
  stepper.setDecelerationInStepsPerSecondPerSecond(DECELERATION_IN_STEPS_PER_SECOND);
  
  // Not start the stepper instance as a service in the "background" as a separate task
  // and the OS of the ESP will take care of invoking the processMovement() task regularily so you can do whatever you want in the loop function
  stepper.startAsService();

  // Last time motor changed initiation
  lastTime = millis();

  // Setup for start/stop button and LED indicator
  pinMode(STOP_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

}

// Main Loop
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** 
void loop() {

  // Get a new normalized sensor event
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  lsm6ds33.getEvent(&accel, &gyro, &temp);

  // Calculate arm angle and convert to degrees
  armAngle = atan2(accel.acceleration.y/GRAVITY, accel.acceleration.x/GRAVITY) * 180 / PI;
  
  Serial.print("Arm angle: ");
  Serial.println(armAngle);

  // Keep moving the relative goal as long as stop isn't pressed, and motorspeed is over zero
  if ((digitalRead(STOP_PIN) == 0) && (motorSpeed > 0))
  {
    stepper.setTargetPositionRelativeInSteps(DISTANCE_TO_TRAVEL_IN_STEPS);
    Serial.println("Motor Running");
    digitalWrite(LED_PIN, HIGH);
  }

  // Debug output
  if (digitalRead(STOP_PIN) == 1) {
    Serial.println("Motor Disabled");
  }
  if (digitalRead(STOP_PIN) == 0) {
    Serial.println("Motor Enabled");
  }

  // If stop pressed, or if motor speed is zero, e-stop the motor
  if ((digitalRead(STOP_PIN) == 1) || (motorSpeed == 0)) {
    stepper.setTargetPositionRelativeInSteps(0);
    stepper.emergencyStop();
    Serial.println("Motor Stopped");
    motorSpeed = 0;
    digitalWrite(LED_PIN, LOW);
  }

  // Check angle and speed every x miliseconds
  if ((millis() - lastTime) > 50) {

    // If arm over critical angle, E-stop
    if (armAngle > 17.5) stepper.emergencyStop();

    // Increase speed or decreas depending on angle
    if (armAngle > 12.5) motorSpeed -= 50;
    else if (armAngle > 7.5)  motorSpeed -= 10;
    else if (armAngle > 5)    motorSpeed -= 5;
    else if (armAngle > 2.5)  motorSpeed -= 2;
    else if (armAngle > 0)    motorSpeed += 0;
    else if (armAngle < 0)    motorSpeed += 2;
    else if (armAngle < 2.5)  motorSpeed += 5;
    else if (armAngle < -5)   motorSpeed += 10;

    // Checks for over or under speed
    if (motorSpeed < 0) motorSpeed = 0;
    if (motorSpeed > MAX_SPEED) motorSpeed = MAX_SPEED;

    Serial.print("Motor speed: ");
    Serial.println(motorSpeed);

    // Sets motor to requested new speed
    stepper.setSpeedInStepsPerSecond(motorSpeed);
  }
  
  delay(50);

}

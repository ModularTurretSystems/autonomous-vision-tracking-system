#include <Servo.h>

Servo pan;
Servo recoil_servo;
Servo flyWheels;

// ====== PAN ======
const int PAN_PIN = 5;
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

const int STEP_DELAY = 15;
const int STEP_SIZE = 1;

int currentAngle = 90;
int targetAngle = 90;
bool isMoving = false;

unsigned long previousMillis = 0;

// ====== ВЫСТРЕЛ ======
const int RECOIL_PIN = 8;
const int FLY_PIN = 9;

const int recoil_rest = 180;
const int recoil_pushed = 120;

// ✅ ИСПРАВЛЕНО (нормальный ESC диапазон)
const int flyWheel_speed = 2000;
const int flyWheel_stop  = 1000;

unsigned long lastShotTime = 0;
const int shotDelay = 1000;

// =======================

void setup() {
  Serial.begin(9600);

  pan.attach(PAN_PIN);
  recoil_servo.attach(RECOIL_PIN);
  flyWheels.attach(FLY_PIN);

  pan.write(currentAngle);

  recoil_servo.write(recoil_rest);

  // 🔥 важно: правильный старт ESC
  flyWheels.writeMicroseconds(flyWheel_stop);
  delay(3000); // дать ESC инициализироваться

  targetAngle = currentAngle;

  Serial.println("=================================");
  Serial.println("    УПРАВЛЕНИЕ СИСТЕМОЙ");
  Serial.println("=================================");
  Serial.println("0-180 - поворот");
  Serial.println("f / fire - выстрел");
  Serial.println("=================================");
}

void loop() {

  // ===== SERIAL =====
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
      processCommand(input);
    }
  }

  // ===== PAN =====
  updateServoPosition();
}

// =======================
// КОМАНДЫ
// =======================
void processCommand(String input) {

  if (input.equalsIgnoreCase("fire") || input.equalsIgnoreCase("f")) {
    Serial.println("🔫 FIRE!");
    shoot();
    return;
  }

  int angle = input.toInt();

  if (angle >= MIN_ANGLE && angle <= MAX_ANGLE) {
    setTargetAngle(angle);
  } else {
    Serial.println("Введите 0-180 или fire");
  }
}

// =======================
// PAN
// =======================
void setTargetAngle(int angle) {
  targetAngle = angle;
  isMoving = true;
}

void updateServoPosition() {

  if (!isMoving || currentAngle == targetAngle) {
    isMoving = false;
    return;
  }

  unsigned long now = millis();

  if (now - previousMillis >= STEP_DELAY) {
    previousMillis = now;

    if (currentAngle < targetAngle)
      currentAngle = min(currentAngle + STEP_SIZE, targetAngle);
    else
      currentAngle = max(currentAngle - STEP_SIZE, targetAngle);

    pan.write(currentAngle);
  }
}

// =======================
// ВЫСТРЕЛ
// =======================
void shoot() {

  // unsigned long now = millis();

  // if (now - lastShotTime < shotDelay) {
  //   Serial.println("⏳ Перезарядка...");
  //   return;
  // }

  // разгон маховиков
  flyWheels.writeMicroseconds(flyWheel_speed);
  delay(400); // время раскрутки

  // выстрел
  recoil_servo.write(recoil_pushed);
  delay(300);
  recoil_servo.write(recoil_rest);

  // lastShotTime = now;
}
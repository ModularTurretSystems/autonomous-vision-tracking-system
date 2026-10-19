#include <Servo.h>

Servo panServo;
Servo tiltServo;
Servo recoilServo;
Servo flyWheels;


// =======================
// КНОПКИ
// =======================
const int BTN_RESET_PIN        = A1;
const int BTN_DISABLE_FIRE_PIN = A2;
const int BTN_3_PIN            = A3;
const int BTN_4_PIN            = A4;

bool fireEnabled = true;

// для отслеживания нажатия
bool lastResetState = HIGH;
bool lastDisableState = HIGH;

// =======================
// ПИНЫ
// =======================
const int PAN_PIN    = 5;
const int TILT_PIN   = 6;
const int RECOIL_PIN = 8;
const int FLY_PIN    = 9;

// =======================
// НАСТРОЙКИ ОСЕЙ
// =======================
const int PAN_CENTER  = 90;
const int TILT_CENTER = 100;

const int PAN_MIN_ANGLE  = 0;
const int PAN_MAX_ANGLE  = 180;
const int TILT_MIN_ANGLE = 80;
const int TILT_MAX_ANGLE = 120;

const int STEP_DELAY_MS = 10;
const int STEP_SIZE     = 5;

// =======================
// ВЫСТРЕЛ (неблокирующий)
// =======================
const int recoil_rest   = 180;
const int recoil_pushed = 100;

const int flyWheel_speed = 2000;
const int flyWheel_stop  = 500;

// Время раскрутки маховиков перед выстрелом (мс)
const unsigned long FLYWHEEL_STARTUP_MS = 400;
// Время удержания отката (мс)
const unsigned long RECOIL_HOLD_MS     = 600;

// Состояния автомата выстрела
enum ShotState {
  SHOT_IDLE,
  SHOT_STARTING_FLYWHEEL,
  SHOT_RECOIL_PUSH,
  SHOT_RECOIL_REST
};

ShotState shotState = SHOT_IDLE;
unsigned long shotTimer = 0;
bool flyRunning = false;

// =======================
// ТЕКУЩИЕ И ЦЕЛЕВЫЕ УГЛЫ
// =======================
int currentPan  = PAN_CENTER;
int targetPan   = PAN_CENTER;

int currentTilt = TILT_CENTER;
int targetTilt  = TILT_CENTER;

unsigned long lastPanStepMs  = 0;
unsigned long lastTiltStepMs = 0;

// =======================
// ПРОТОТИПЫ
// =======================
void processCommand(String input);
bool extractSignedValue(const String& input, const String& key, int& value);
void updatePanPosition();
void updateTiltPosition();
void updateShot();          // новая функция для неблокирующего выстрела
void startFlyWheels();
void stopFlyWheels();
void shoot();               // теперь только запускает автомат

void updateButtons();

// =======================
// SETUP
// =======================
void setup() {
  Serial.begin(9600);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  recoilServo.attach(RECOIL_PIN);
  flyWheels.attach(FLY_PIN);

  pinMode(BTN_RESET_PIN, INPUT_PULLUP);
  pinMode(BTN_DISABLE_FIRE_PIN, INPUT_PULLUP);
  pinMode(BTN_3_PIN, INPUT_PULLUP);
  pinMode(BTN_4_PIN, INPUT_PULLUP);

  panServo.write(currentPan);
  tiltServo.write(currentTilt);
  recoilServo.write(recoil_rest);

  flyWheels.writeMicroseconds(flyWheel_stop);
  delay(3000); // инициализация ESC

  Serial.println("=================================");
  Serial.println("   TURRET READY (non-blocking)");
  Serial.println("=================================");
  Serial.println("Format: PAN=-5, TILT=-12");
  Serial.println("Fire: f / fire");
  Serial.println("Stop flywheels: stop / s");
  Serial.println("Start flywheels: start");
  Serial.println("=================================");
}

// =======================
// LOOP
// =======================
void loop() {
  updateButtons();
  // Чтение команд с компьютера
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      processCommand(input);
    }
  }

  // Движение серв (неблокирующее)
  updatePanPosition();
  updateTiltPosition();

  // Обработка выстрела (неблокирующая)
  updateShot();
}

void updateButtons() {

  bool resetPressed =
      digitalRead(BTN_RESET_PIN);

  bool disablePressed =
      digitalRead(BTN_DISABLE_FIRE_PIN);

  // =================================
  // КНОПКА 1 -> СБРОС ПОЗИЦИИ
  // =================================
  if (lastResetState == HIGH &&
      resetPressed == LOW) {

    targetPan  = PAN_CENTER;
    targetTilt = TILT_CENTER;

    Serial.println("🔄 Turret reset to center");
  }

  // =================================
  // КНОПКА 2 -> TOGGLE FIRE
  // =================================
  if (lastDisableState == HIGH &&
      disablePressed == LOW) {

    fireEnabled = !fireEnabled;

    Serial.print("🔥 Fire enabled: ");
    Serial.println(fireEnabled ? "TRUE" : "FALSE");
  }

  lastResetState   = resetPressed;
  lastDisableState = disablePressed;
}

// =======================
// ПАРСИНГ КОМАНД
// =======================
void processCommand(String input) {
  input.trim();
  input.toUpperCase();

  if (input == "F" || input == "FIRE") {
    shoot();
    return;
  }

  if (input == "STOP" || input == "S") {
    stopFlyWheels();
    return;
  }

  if (input == "START") {
    startFlyWheels();
    return;
  }

  // =====================================
  // АБСОЛЮТНЫЕ УГЛЫ
  // =====================================
  int setPan = 0;
  int setTilt = 0;

  bool hasSetPan =
      extractSignedValue(input, "SETPAN", setPan);

  bool hasSetTilt =
      extractSignedValue(input, "SETTILT", setTilt);

  if (hasSetPan || hasSetTilt) {

    if (hasSetPan) {
      targetPan = constrain(
          setPan,
          PAN_MIN_ANGLE,
          PAN_MAX_ANGLE
      );
    }

    if (hasSetTilt) {
      targetTilt = constrain(
          setTilt,
          TILT_MIN_ANGLE,
          TILT_MAX_ANGLE
      );
    }

    Serial.print("ABS TARGET -> ");
    Serial.print("PAN=");
    Serial.print(targetPan);
    Serial.print(" TILT=");
    Serial.println(targetTilt);

    return;
  }

  // =====================================
  // ОТНОСИТЕЛЬНЫЕ УГЛЫ (старый режим)
  // =====================================
  int panOffset = 0;
  int tiltOffset = 0;

  bool hasPan =
      extractSignedValue(input, "PAN", panOffset);

  bool hasTilt =
      extractSignedValue(input, "TILT", tiltOffset);

  if (hasPan || hasTilt) {

    if (hasPan) {
      targetPan = constrain(
          currentPan + panOffset,
          PAN_MIN_ANGLE,
          PAN_MAX_ANGLE
      );
    }

    if (hasTilt) {
      targetTilt = constrain(
          currentTilt + tiltOffset,
          TILT_MIN_ANGLE,
          TILT_MAX_ANGLE
      );
    }

    Serial.print("REL TARGET -> ");
    Serial.print("PAN=");
    Serial.print(targetPan);
    Serial.print(" TILT=");
    Serial.println(targetTilt);

    return;
  }

  Serial.print("Unknown command: ");
  Serial.println(input);
}

bool extractSignedValue(const String& input, const String& key, int& value) {
  int idx = input.indexOf(key);
  if (idx == -1) return false;

  idx += key.length();
  while (idx < input.length()) {
    char c = input.charAt(idx);
    if (c == ' ' || c == '\t' || c == '=' || c == ':') idx++;
    else break;
  }

  int end = idx;
  bool foundDigit = false;
  while (end < input.length()) {
    char c = input.charAt(end);
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      foundDigit = true;
      end++;
    } else break;
  }

  if (!foundDigit) return false;
  value = input.substring(idx, end).toInt();
  return true;
}

// =======================
// ДВИЖЕНИЕ PAN
// =======================
void updatePanPosition() {
  if (currentPan == targetPan) return;

  unsigned long now = millis();
  if (now - lastPanStepMs < STEP_DELAY_MS) return;
  lastPanStepMs = now;

  if (currentPan < targetPan)
    currentPan = min(currentPan + STEP_SIZE, targetPan);
  else
    currentPan = max(currentPan - STEP_SIZE, targetPan);

  panServo.write(currentPan);
}

// =======================
// ДВИЖЕНИЕ TILT
// =======================
void updateTiltPosition() {
  if (currentTilt == targetTilt) return;

  unsigned long now = millis();
  if (now - lastTiltStepMs < STEP_DELAY_MS) return;
  lastTiltStepMs = now;

  if (currentTilt < targetTilt)
    currentTilt = min(currentTilt + STEP_SIZE, targetTilt);
  else
    currentTilt = max(currentTilt - STEP_SIZE, targetTilt);

  tiltServo.write(currentTilt);
}

// =======================
// НЕБЛОКИРУЮЩИЙ ВЫСТРЕЛ
// =======================
void shoot() {

  if (!fireEnabled) {
    Serial.println("❌ Fire disabled");
    return;
  }

  // Запускаем автомат, только если он в покое
  if (shotState == SHOT_IDLE) {
    shotState = SHOT_STARTING_FLYWHEEL;
    shotTimer = millis();
    startFlyWheels();

    Serial.println("🔫 Fire sequence started");

  } else {
    Serial.println("❌ Busy, can't fire now");
  }
}

void updateShot() {
  if (shotState == SHOT_IDLE) return;

  unsigned long now = millis();

  switch (shotState) {
    case SHOT_STARTING_FLYWHEEL:
      // Ждём, пока маховики раскрутятся
      if (now - shotTimer >= FLYWHEEL_STARTUP_MS) {
        // Включаем откат
        recoilServo.write(recoil_pushed);
        shotState = SHOT_RECOIL_PUSH;
        shotTimer = now;
        // Serial.println("  → Recoil push");
      }
      break;

    case SHOT_RECOIL_PUSH:
      // Держим откат указанное время
      if (now - shotTimer >= RECOIL_HOLD_MS) {
        // Возвращаем откат в исходное положение
        recoilServo.write(recoil_rest);
        shotState = SHOT_RECOIL_REST;
        shotTimer = now;
        // Serial.println("  → Recoil return");
      }
      break;

    case SHOT_RECOIL_REST:
      // Дадим серве отката время физически вернуться (необязательная задержка)
      // Можно сделать небольшую паузу (например, 50 мс) или сразу в IDLE.
      // Здесь сразу заканчиваем, но если нужна "перезарядка" – добавьте состояние.
      shotState = SHOT_IDLE;
      // Serial.println("✅ Fire sequence finished");
      break;
  }
}

// =======================
// УПРАВЛЕНИЕ МАХОВИКАМИ
// =======================
void startFlyWheels() {
  if (!flyRunning) {
    Serial.println("🌀 Flywheels START");
    flyWheels.writeMicroseconds(flyWheel_speed);
    flyRunning = true;
  }
}

void stopFlyWheels() {
  if (flyRunning) {
    Serial.println("🛑 Flywheels STOP");
    flyWheels.writeMicroseconds(flyWheel_stop);
    flyRunning = false;
  }
}

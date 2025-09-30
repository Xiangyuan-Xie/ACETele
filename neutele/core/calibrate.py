from core.integrate import TeleCore


def main():
    tele_core = TeleCore()
    result = tele_core.calibrate()
    if result:
        print(f"标定成功，当前姿态为{tele_core.act()}")
    else:
        print("标定失败！")


if __name__ == "__main__":
    main()

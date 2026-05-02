import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  // ✅ Brand (keep same names so old code works)
  // A richer dark green + fresh mint (still your identity)
  static const Color darkGreen = Color(0xFF06261E);
  static const Color primary = Color(0xFF35D6A6);

  // ✅ Backgrounds / Surfaces (modern soft)
  static const Color surface = Color(0xFFF4FBF8);
  static const Color textDark = Color(0xFF071E17);
  static const Color textLight = Color(0xFFE9FFF6);

  // ✅ Border (more consistent + clean)
  static const Color border = Color(0x2206261E);

  // ✅ Status / Error
  static const Color error = Color(0xFFD64545);

  // ✅ Extra useful modern colors (optional, won’t break anything)
  static const Color card = Color(0xFFFFFFFF);
  static const Color muted = Color(0xFF5B6B66);
  static const Color shadow = Color(0x14000000);

  // ✅ A more premium hero gradient (still green-based)
  static const LinearGradient heroGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFF06261E),
      Color(0xFF0C3F33),
      Color(0xFF1B7D63),
    ],
  );
}

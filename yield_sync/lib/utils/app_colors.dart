import 'package:flutter/material.dart';

class AppColors {
  AppColors._();

  // Plant-shop inspired palette (soft green + clean neutrals)
  static const Color darkGreen = Color(0xFF2D5A27);
  static const Color primary = Color(0xFF8DCB5A);

  static const Color surface = Color(0xFFF7FAF3);
  static const Color textDark = Color(0xFF1F2A1F);
  static const Color textLight = Color(0xFFF6FFF0);

  static const Color border = Color(0x223C5235);

  // ✅ Status / Error
  static const Color error = Color(0xFFD64545);

  static const Color card = Color(0xFFFFFFFF);
  static const Color muted = Color(0xFF627062);
  static const Color shadow = Color(0x14000000);

  static const LinearGradient heroGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFF4F8B3E),
      Color(0xFF77B850),
      Color(0xFF9ED06E),
    ],
  );
}

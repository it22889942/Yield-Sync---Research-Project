import 'package:flutter/material.dart';
import 'app_colors.dart';

class AppTheme {
  AppTheme._();

  static ThemeData get light {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      fontFamily: null,
    );

    final cs = ColorScheme.fromSeed(
      seedColor: AppColors.primary,
      brightness: Brightness.light,
      primary: AppColors.primary,
      error: AppColors.error,
      surface: Colors.white,
    );

    return base.copyWith(
      colorScheme: cs,

      scaffoldBackgroundColor: AppColors.surface,

      // ✅ AppBar (clean modern)
      appBarTheme: AppBarTheme(
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        backgroundColor: Colors.transparent,
        foregroundColor: AppColors.textDark,
        titleTextStyle: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w900,
          color: AppColors.textDark,
        ),
      ),

      chipTheme: ChipThemeData(
        backgroundColor: Colors.white,
        selectedColor: AppColors.primary.withOpacity(0.2),
        side: const BorderSide(color: AppColors.border),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        labelStyle: const TextStyle(
          fontWeight: FontWeight.w800,
          color: AppColors.textDark,
        ),
      ),

      // ✅ Text (better hierarchy / readability)
      textTheme: base.textTheme.copyWith(
        titleLarge: const TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w900,
          color: AppColors.textDark,
          letterSpacing: -0.2,
        ),
        titleMedium: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w900,
          color: AppColors.textDark,
        ),
        bodyLarge: TextStyle(
          fontSize: 14.5,
          fontWeight: FontWeight.w700,
          color: AppColors.textDark.withOpacity(0.90),
        ),
        bodyMedium: TextStyle(
          fontSize: 13.5,
          fontWeight: FontWeight.w600,
          color: AppColors.textDark.withOpacity(0.78),
        ),
        labelLarge: const TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w900,
        ),
      ),

      // ✅ Cards (modern rounded + soft shadow)
      cardTheme: CardThemeData(
        color: AppColors.card,
        elevation: 0,
        shadowColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: const BorderSide(color: AppColors.border),
        ),
        margin: EdgeInsets.zero,
      ),

      // ✅ Dividers
      dividerTheme: DividerThemeData(
        color: AppColors.border.withOpacity(0.75),
        thickness: 1,
        space: 1,
      ),

      // ✅ Input fields (premium)
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        hintStyle: TextStyle(
          color: AppColors.textDark.withOpacity(0.45),
          fontWeight: FontWeight.w600,
        ),
        labelStyle: TextStyle(
          color: AppColors.textDark.withOpacity(0.80),
          fontWeight: FontWeight.w800,
        ),
        prefixIconColor: AppColors.textDark.withOpacity(0.60),
        suffixIconColor: AppColors.textDark.withOpacity(0.60),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(
              color: AppColors.primary.withOpacity(0.95), width: 1.9),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.error, width: 1.6),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.error, width: 1.9),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      ),

      // ✅ Elevated buttons (pill + nice press feel)
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          elevation: 0,
          backgroundColor: AppColors.primary,
          foregroundColor: AppColors.darkGreen,
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(999),
          ),
          textStyle: const TextStyle(
            fontWeight: FontWeight.w900,
            fontSize: 14.5,
          ),
        ).copyWith(
          overlayColor: MaterialStateProperty.all(
            AppColors.darkGreen.withOpacity(0.08),
          ),
        ),
      ),

      // ✅ Outlined buttons (clean)
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.textDark,
          side: const BorderSide(color: AppColors.border),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          textStyle: const TextStyle(fontWeight: FontWeight.w900),
        ).copyWith(
          overlayColor: MaterialStateProperty.all(
            AppColors.primary.withOpacity(0.10),
          ),
        ),
      ),

      // ✅ Text buttons
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: AppColors.textDark,
          textStyle: const TextStyle(fontWeight: FontWeight.w900),
        ).copyWith(
          overlayColor: MaterialStateProperty.all(
            AppColors.primary.withOpacity(0.10),
          ),
        ),
      ),

      // ✅ Snackbars (modern floating)
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.textDark,
        contentTextStyle: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w700,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),

      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: Colors.white,
        selectedItemColor: AppColors.darkGreen,
        unselectedItemColor: AppColors.muted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
        selectedLabelStyle: const TextStyle(fontWeight: FontWeight.w800),
        unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w700),
      ),

      // ✅ Dialogs (your dialogs will look nicer automatically)
      dialogTheme: DialogThemeData(
        backgroundColor: const Color(0xFFF2EEF6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
        ),
        titleTextStyle: const TextStyle(
          fontWeight: FontWeight.w900,
          fontSize: 18,
          color: AppColors.textDark,
        ),
        contentTextStyle: TextStyle(
          fontWeight: FontWeight.w700,
          color: AppColors.textDark.withOpacity(0.78),
        ),
      ),
    );
  }
}

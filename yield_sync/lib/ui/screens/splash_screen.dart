import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/nav.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: AppColors.heroGradient,
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
            child: Column(
              children: [
                // Top branding (like app header)
                Row(
                  children: [
                    Container(
                      height: 44,
                      width: 44,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(12),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.14)),
                      ),
                      child: const Icon(Icons.eco_rounded,
                          color: AppColors.primary),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      "YieldSync",
                      style: TextStyle(
                        color: AppColors.textLight.withOpacity(0.95),
                        fontWeight: FontWeight.w800,
                        fontSize: 16,
                        letterSpacing: 0.2,
                      ),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(999),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.14)),
                      ),
                      child: Text(
                        "Sri Lanka",
                        style: TextStyle(
                          color: AppColors.textLight.withOpacity(0.9),
                          fontWeight: FontWeight.w700,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),

                // ✅ This keeps the card perfectly centered
                const Expanded(child: SizedBox()),

                // Center Welcome Card (modern UI)
                Container(
                  width: double.infinity,
                  constraints: const BoxConstraints(maxWidth: 440),
                  padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: Colors.white.withOpacity(0.14)),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.18),
                        blurRadius: 28,
                        offset: const Offset(0, 18),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      // LOGO card
                      Container(
                        width: 94,
                        height: 94,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(24),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Image.asset(
                            "assets/images/logo.jpg",
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        "Welcome",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: AppColors.textLight.withOpacity(0.98),
                          fontSize: 38,
                          fontWeight: FontWeight.w900,
                          height: 1.05,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "Find skilled workers & manage\nyour workflow with YieldSync.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: AppColors.textLight.withOpacity(0.78),
                          fontSize: 14.5,
                          height: 1.35,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 18),
                      SizedBox(
                        width: size.width * 0.70,
                        height: 52,
                        child: ElevatedButton(
                          onPressed: () =>
                              Nav.replace(context, AppRoutes.login),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            foregroundColor: AppColors.darkGreen,
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(18),
                            ),
                          ),
                          child: const Text(
                            "Let’s get started",
                            style: TextStyle(
                              fontSize: 15.5,
                              fontWeight: FontWeight.w900,
                              letterSpacing: 0.2,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                // ✅ This balances the screen so it stays centered
                const Expanded(child: SizedBox()),

                // Bottom items stay at bottom
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _dot(active: true),
                    const SizedBox(width: 8),
                    _dot(active: false),
                    const SizedBox(width: 8),
                    _dot(active: false),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  "v1.0 UI Stage",
                  style: TextStyle(
                    color: AppColors.textLight.withOpacity(0.55),
                    fontWeight: FontWeight.w600,
                    fontSize: 12.5,
                  ),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }

  static Widget _dot({required bool active}) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: active ? 28 : 10,
      height: 10,
      decoration: BoxDecoration(
        color: active ? AppColors.primary : Colors.white.withOpacity(0.25),
        borderRadius: BorderRadius.circular(999),
      ),
    );
  }
}

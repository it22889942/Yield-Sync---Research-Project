import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/nav.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9FDF2),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final w = constraints.maxWidth;
            final s = (w / 430).clamp(0.86, 1.06);

            return Padding(
              padding: EdgeInsets.fromLTRB(20 * s, 10 * s, 20 * s, 16 * s),
              child: Column(
                children: [
                  SizedBox(height: 18 * s),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Container(
                      padding:
                          EdgeInsets.symmetric(horizontal: 12 * s, vertical: 6 * s),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.18),
                        borderRadius: BorderRadius.circular(999),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: Text(
                        "SMART FARM PLATFORM",
                        style: TextStyle(
                          color: AppColors.darkGreen,
                          fontSize: 10.5 * s,
                          letterSpacing: 1.0,
                          fontWeight: FontWeight.w700,
                        ).merge(GoogleFonts.inter()),
                      ),
                    ),
                  ),
                  SizedBox(height: 14 * s),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Farm Better With",
                          style: GoogleFonts.poppins(
                            fontSize: 34 * s,
                            fontWeight: FontWeight.w500,
                            color: AppColors.textDark,
                            height: 1.06,
                            letterSpacing: -0.2,
                          ),
                        ),
                        Text(
                          "YieldSync.",
                          style: GoogleFonts.poppins(
                            fontSize: 34 * s,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark,
                            height: 1.02,
                            letterSpacing: -0.2,
                          ),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(height: 8 * s),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      "One app for labour hiring, equipment rentals,\nand data-driven farming decisions.",
                      style: GoogleFonts.inter(
                        color: AppColors.textDark.withOpacity(0.68),
                        fontSize: 13.4 * s,
                        height: 1.4,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  SizedBox(height: 26 * s),
                  Expanded(
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        Positioned(
                          left: 6 * s,
                          top: 102 * s,
                          child: RotatedBox(
                            quarterTurns: 3,
                            child: Text(
                              "YIELD SYNC",
                              style: GoogleFonts.inter(
                                color: AppColors.textDark.withOpacity(0.85),
                                fontSize: 10.5 * s,
                                letterSpacing: 2.0,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ),
                        Positioned(
                          top: 10 * s,
                          child: SizedBox(
                            width: 350 * s,
                            height: 350 * s,
                            child: Image.asset(
                              "assets/images/farmer-splash.png",
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),
                        Positioned(
                          bottom: 22 * s,
                          child: Container(
                            width: 292 * s,
                            height: 76 * s,
                            decoration: BoxDecoration(
                              color: AppColors.surface,
                              borderRadius: BorderRadius.circular(160 * s),
                              border: Border.all(
                                color: AppColors.primary.withOpacity(0.18),
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: AppColors.darkGreen.withOpacity(0.06),
                                  blurRadius: 18 * s,
                                  offset: Offset(0, 6 * s),
                                ),
                              ],
                            ),
                          ),
                        ),
                        Positioned(
                          bottom: 40 * s,
                          left: 0,
                          right: 0,
                          child: Center(
                            child: Container(
                              padding: EdgeInsets.symmetric(
                                horizontal: 12 * s,
                                vertical: 6 * s,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(color: AppColors.border),
                              ),
                              child: Text(
                                "Empowered Happy Farmers",
                                textAlign: TextAlign.center,
                                style: GoogleFonts.poppins(
                                  fontSize: 13.5 * s,
                                  fontWeight: FontWeight.w500,
                                  fontStyle: FontStyle.italic,
                                  color: AppColors.darkGreen.withOpacity(0.82),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(height: 10 * s),
                  Text(
                    "Trusted by modern farmers to simplify\ndaily operations from one place.",
                    textAlign: TextAlign.center,
                    style: GoogleFonts.inter(
                      color: AppColors.textDark.withOpacity(0.88),
                      fontSize: 17 * s,
                      height: 1.4,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  SizedBox(height: 22 * s),
                  InkWell(
                    onTap: () => Nav.replace(context, AppRoutes.login),
                    borderRadius: BorderRadius.circular(999),
                    child: Container(
                      width: 104 * s,
                      height: 104 * s,
                      padding: EdgeInsets.all(6 * s),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.24),
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: AppColors.darkGreen.withOpacity(0.14),
                            blurRadius: 18 * s,
                            offset: Offset(0, 8 * s),
                          ),
                        ],
                      ),
                      child: Container(
                        decoration: BoxDecoration(
                          color: AppColors.primary,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: AppColors.darkGreen.withOpacity(0.14),
                            width: 2.2 * s,
                          ),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              "ENTER",
                              style: GoogleFonts.poppins(
                                fontWeight: FontWeight.w700,
                                color: AppColors.darkGreen,
                                fontSize: 13.2 * s,
                                letterSpacing: 0.7,
                              ),
                            ),
                            SizedBox(height: 2 * s),
                            // Icon(
                            //   Icons.arrow_forward_rounded,
                            //   size: 17 * s,
                            //   color: AppColors.darkGreen.withOpacity(0.9),
                            // ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: 6 * s),
                  Text(
                    "Tap to continue",
                    style: GoogleFonts.inter(
                      color: AppColors.textDark.withOpacity(0.5),
                      fontSize: 11.5 * s,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.2,
                    ),
                  ),
                  SizedBox(height: 8 * s),
                  Text(
                    "Fast setup. Real-time access. Better outcomes.",
                    style: GoogleFonts.inter(
                      color: AppColors.textDark.withOpacity(0.58),
                      fontSize: 11.8 * s,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  SizedBox(height: 8 * s),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

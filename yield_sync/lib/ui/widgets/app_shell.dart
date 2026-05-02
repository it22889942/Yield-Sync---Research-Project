import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';

class AppShell extends StatelessWidget {
  final int currentIndex; // keep (0=Home) for compatibility
  final Widget child;

  const AppShell({
    super.key,
    required this.currentIndex,
    required this.child,
  });

  void _goHome(BuildContext context) {
    Navigator.pushNamedAndRemoveUntil(
      context,
      AppRoutes.home,
      (route) => false,
      arguments: 0,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(child: child),

      // ✅ Home-style bottom pill bar (ONLY Home)
      bottomNavigationBar: SafeArea(
        top: false,
        child: SizedBox(
          height: 64,
          child: BottomAppBar(
            color: Colors.white,
            elevation: 0,
            padding: EdgeInsets.zero,
            child: Center(
              child: Material(
                color: AppColors.primary.withOpacity(0.12),
                shape: StadiumBorder(
                  side: BorderSide(color: AppColors.border),
                ),
                child: InkWell(
                  customBorder: const StadiumBorder(),
                  onTap: () => _goHome(context),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 18, vertical: 10),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.home_rounded,
                          color: AppColors.darkGreen,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          "Home",
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark.withOpacity(0.90),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

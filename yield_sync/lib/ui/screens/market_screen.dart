import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';

class MarketScreen extends StatelessWidget {
  const MarketScreen({super.key});

  // ✅ image asset path
  static const String _marketImage = "assets/images/paddymarket.jpeg";

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0, // keep Home active
      child: Column(
        children: [
          // ===== HEADER (modern) =====
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            decoration: const BoxDecoration(
              gradient: AppColors.heroGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(28),
                bottomRight: Radius.circular(28),
              ),
            ),
            child: Stack(
              children: [
                const Positioned(
                  right: -14,
                  top: 16,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.show_chart_rounded,
                        size: 150, color: Colors.white),
                  ),
                ),
                const Positioned(
                  left: -16,
                  bottom: -18,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.storefront_rounded,
                        size: 140, color: Colors.white),
                  ),
                ),
                Column(
                  children: [
                    Row(
                      children: [
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.arrow_back_ios_new_rounded),
                          color: Colors.white,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          "Market",
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.92),
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),
                        InkWell(
                          borderRadius: BorderRadius.circular(999),
                          onTap: () =>
                              Navigator.pushNamed(context, AppRoutes.profile),
                          child: CircleAvatar(
                            radius: 18,
                            backgroundColor: Colors.white.withOpacity(0.12),
                            child: const Icon(Icons.person_rounded,
                                color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Price ",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.98),
                        fontWeight: FontWeight.w900,
                        fontSize: 22,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      "Forecast • Analytics • History",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.78),
                        fontWeight: FontWeight.w700,
                        fontSize: 12.8,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // ✅ small “info chips”
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      alignment: WrapAlignment.center,
                      children: const [
                        _HeaderChip(
                            icon: Icons.trending_up_rounded, text: "Trends"),
                        _HeaderChip(
                            icon: Icons.insights_rounded, text: "Analytics"),
                        _HeaderChip(
                            icon: Icons.history_rounded, text: "Compare"),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 14),

          // ===== BODY =====
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              children: [
                // ✅ IMAGE CARD (asset image)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: AppColors.border),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 18,
                        offset: const Offset(0, 12),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      // ✅ Replaced placeholder with image
                      ClipRRect(
                        borderRadius: BorderRadius.circular(22),
                        child: SizedBox(
                          width: 150,
                          height: 150,
                          child: Image.asset(
                            _marketImage,
                            fit: BoxFit.cover,
                            errorBuilder: (_, __, ___) => Container(
                              color: AppColors.primary.withOpacity(0.12),
                              child: const Icon(
                                Icons.image_not_supported_rounded,
                                color: AppColors.darkGreen,
                                size: 34,
                              ),
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "Smart Market Forecast",
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                fontSize: 15.5,
                                color: AppColors.textDark,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              "Choose crop & market to see\nprice insights.",
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                color: AppColors.textDark.withOpacity(0.62),
                                height: 1.25,
                              ),
                            ),
                            const SizedBox(height: 10),
                            Row(
                              children: const [
                                _MiniPill(
                                    icon: Icons.timer_rounded,
                                    text: "7/14/30 days"),
                                SizedBox(width: 8),
                                _MiniPill(
                                    icon: Icons.security_rounded,
                                    text: "Saved history"),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 14),

                _MarketActionCard(
                  title: "Forecast Input",
                  subtitle: "Select crop, market & horizon",
                  icon: Icons.edit_note_rounded,
                  onTap: () => Navigator.pushNamed(
                    context,
                    AppRoutes.marketForecastInput,
                  ),
                ),
                const SizedBox(height: 12),

                _MarketActionCard(
                  title: "Comparison & History",
                  subtitle: "Compare markets and previous runs",
                  icon: Icons.compare_arrows_rounded,
                  onTap: () => Navigator.pushNamed(
                    context,
                    AppRoutes.marketComparisonHistory,
                  ),
                ),
                const SizedBox(height: 12),

                _MarketActionCard(
                  title: "Update Data & Train",
                  subtitle: "Fetch latest week and retrain models",
                  icon: Icons.sync_alt_rounded,
                  onTap: () => Navigator.pushNamed(
                    context,
                    AppRoutes.marketDataUpdate,
                  ),
                ),

                const SizedBox(height: 14),

                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: AppColors.border),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 38,
                        height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Icon(Icons.lightbulb_rounded,
                            color: AppColors.textDark.withOpacity(0.75)),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          "Tip: Try 7-day forecast for quick selling decisions.",
                          style: TextStyle(
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark.withOpacity(0.70),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ===================== SMALL WIDGETS =====================

class _HeaderChip extends StatelessWidget {
  final IconData icon;
  final String text;

  const _HeaderChip({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white.withOpacity(0.95)),
          const SizedBox(width: 8),
          Text(
            text,
            style: TextStyle(
              color: Colors.white.withOpacity(0.92),
              fontWeight: FontWeight.w800,
              fontSize: 12.2,
            ),
          ),
        ],
      ),
    );
  }
}

class _MiniPill extends StatelessWidget {
  final IconData icon;
  final String text;

  const _MiniPill({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: AppColors.darkGreen),
          const SizedBox(width: 7),
          Text(
            text,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              color: AppColors.textDark,
              fontSize: 12.1,
            ),
          ),
        ],
      ),
    );
  }
}

class _MarketActionCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final VoidCallback onTap;

  const _MarketActionCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(22),
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: AppColors.border),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.04),
                blurRadius: 14,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(icon, color: AppColors.darkGreen, size: 26),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        fontSize: 14.6,
                        color: AppColors.textDark,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: AppColors.textDark.withOpacity(0.62),
                        height: 1.2,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.arrow_forward_ios_rounded,
                  size: 16, color: AppColors.textDark.withOpacity(0.35)),
            ],
          ),
        ),
      ),
    );
  }
}

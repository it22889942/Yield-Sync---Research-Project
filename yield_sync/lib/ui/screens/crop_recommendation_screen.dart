import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';

class CropRecommendationScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const CropRecommendationScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final title = (data["name"] ?? data["docId"] ?? "Crop").toString();
    final imageAsset = (data["imageAsset"] ?? "").toString();

    final seasonal = (data["seasonalFactors"] is List)
        ? List<String>.from(data["seasonalFactors"])
        : <String>[];

    final zones =
        (data["zones"] is List) ? List<String>.from(data["zones"]) : <String>[];

    final climate = (data["climate"] ?? "").toString();

    final fert = (data["fertilizerPerHectare"] is Map)
        ? Map<String, dynamic>.from(data["fertilizerPerHectare"])
        : <String, dynamic>{};

    final cultivation = (data["cultivationProcess"] ?? "").toString();

    List<String> _list(dynamic v) =>
        (v is List) ? v.map((e) => e.toString()).toList() : <String>[];

    final basal = _list(fert["basal"]);
    final top = _list(fert["topDressing"]);
    final optional = _list(fert["optional"]);
    final tips = _list(fert["tips"]);

    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          _HeroHeader(
            title: title,
            imageAsset: imageAsset,
            onBack: () => Navigator.pop(context),
          ),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
              children: [
                Row(
                  children: [
                    Expanded(
                      child: _MiniStatCard(
                        icon: Icons.public_rounded,
                        title: "Zones",
                        value: zones.isEmpty ? "—" : "${zones.length} areas",
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _MiniStatCard(
                        icon: Icons.calendar_month_rounded,
                        title: "Season",
                        value: seasonal.isEmpty
                            ? "—"
                            : "${seasonal.length} factors",
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                if (zones.isNotEmpty)
                  _SectionCard(
                    icon: Icons.location_on_rounded,
                    title: "Zones",
                    child: Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: zones.map((z) => _Chip(text: z)).toList(),
                    ),
                  ),
                if (climate.trim().isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    icon: Icons.cloud_rounded,
                    title: "Climate",
                    child: Text(
                      climate,
                      style: TextStyle(
                        color: AppColors.textDark.withOpacity(0.80),
                        fontWeight: FontWeight.w700,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
                if (seasonal.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    icon: Icons.wb_sunny_rounded,
                    title: "Seasonal Factors",
                    child: Column(
                      children: seasonal
                          .map((t) => _Bullet(text: t))
                          .toList(growable: false),
                    ),
                  ),
                ],
                if (basal.isNotEmpty ||
                    top.isNotEmpty ||
                    optional.isNotEmpty ||
                    tips.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _FertilizerModernCard(
                    basal: basal,
                    top: top,
                    optional: optional,
                    tips: tips,
                  ),
                ],
                if (cultivation.trim().isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    icon: Icons.grass_rounded,
                    title: "Cultivation Process",
                    child: Text(
                      cultivation,
                      style: TextStyle(
                        color: AppColors.textDark.withOpacity(0.80),
                        fontWeight: FontWeight.w700,
                        height: 1.45,
                      ),
                    ),
                  ),
                ],
                const SizedBox(height: 6),
                // Container(
                //   padding:
                //       const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                //   decoration: BoxDecoration(
                //     color: AppColors.primary.withOpacity(0.10),
                //     borderRadius: BorderRadius.circular(16),
                //     border: Border.all(color: AppColors.border),
                //   ),
                //   child: Row(
                //     children: [
                //       const Icon(Icons.info_outline_rounded,
                //           color: AppColors.darkGreen),
                //       const SizedBox(width: 10),
                //       Expanded(
                //         child: Text(
                //           "Tip: Follow fertilizer steps exactly and adjust based on soil test results.",
                //           style: TextStyle(
                //             color: AppColors.textDark.withOpacity(0.78),
                //             fontWeight: FontWeight.w700,
                //           ),
                //         ),
                //       ),
                //     ],
                //   ),
                // ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/* =========================
   HERO HEADER (modern)
========================= */

class _HeroHeader extends StatelessWidget {
  final String title;
  final String imageAsset;
  final VoidCallback onBack;

  const _HeroHeader({
    required this.title,
    required this.imageAsset,
    required this.onBack,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Column(
          children: [
            Row(
              children: [
                IconButton(
                  onPressed: onBack,
                  icon: const Icon(Icons.arrow_back_ios_new_rounded),
                  color: Colors.white,
                ),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.94),
                      fontWeight: FontWeight.w900,
                      fontSize: 18,
                      letterSpacing: 0.2,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                CircleAvatar(
                  radius: 18,
                  backgroundColor: Colors.white.withOpacity(0.14),
                  child: const Icon(Icons.eco_rounded, color: Colors.white),
                ),
              ],
            ),
            const SizedBox(height: 10),

            // ✅ Hero image (tap to zoom)
            Container(
              height: 150,
              width: double.infinity,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: Colors.white.withOpacity(0.16)),
                color: Colors.white.withOpacity(0.08),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(22),
                child: imageAsset.isNotEmpty
                    ? _ZoomableHeroImage(imageAsset: imageAsset)
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.image_not_supported_rounded,
                                color: Colors.white.withOpacity(0.65)),
                            const SizedBox(height: 6),
                            Text(
                              "No image",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.72),
                                fontWeight: FontWeight.w800,
                              ),
                            )
                          ],
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/* =========================
   ZOOMABLE HERO IMAGE (popup)
========================= */

class _ZoomableHeroImage extends StatelessWidget {
  final String imageAsset;
  const _ZoomableHeroImage({required this.imageAsset});

  void _open(BuildContext context) {
    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: "Image",
      barrierColor: Colors.black.withOpacity(0.85),
      pageBuilder: (_, __, ___) {
        return SafeArea(
          child: Scaffold(
            backgroundColor: Colors.transparent,
            body: Stack(
              children: [
                Center(
                  child: InteractiveViewer(
                    minScale: 1,
                    maxScale: 5,
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: Image.asset(imageAsset, fit: BoxFit.contain),
                    ),
                  ),
                ),
                Positioned(
                  top: 10,
                  left: 10,
                  child: _RoundIconBtn(
                    icon: Icons.close_rounded,
                    onTap: () => Navigator.pop(context),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _open(context),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.asset(imageAsset, fit: BoxFit.cover),

          // overlay (modern)
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.black.withOpacity(0.08),
                  Colors.black.withOpacity(0.42),
                ],
              ),
            ),
          ),

          // bottom hint
          Align(
            alignment: Alignment.bottomLeft,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(color: Colors.white.withOpacity(0.25)),
                    ),
                    child: Text(
                      "Tap to view",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.95),
                        fontWeight: FontWeight.w900,
                        fontSize: 12,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    width: 34,
                    height: 34,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white.withOpacity(0.25)),
                    ),
                    child: const Icon(Icons.zoom_in_rounded,
                        color: Colors.white, size: 18),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RoundIconBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const _RoundIconBtn({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white.withOpacity(0.14),
      shape: const CircleBorder(),
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Icon(icon, color: Colors.white, size: 22),
        ),
      ),
    );
  }
}

/* =========================
   SECTION CARD
========================= */

class _SectionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final Widget child;

  const _SectionCard({
    required this.icon,
    required this.title,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 34,
                height: 34,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(icon, color: AppColors.darkGreen, size: 18),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 15.5,
                    color: AppColors.textDark,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}

/* =========================
   MINI STAT CARD
========================= */

class _MiniStatCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;

  const _MiniStatCard({
    required this.icon,
    required this.title,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 16,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.14),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: AppColors.textDark.withOpacity(0.65),
                    fontWeight: FontWeight.w900,
                    fontSize: 12.5,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: const TextStyle(
                    color: AppColors.textDark,
                    fontWeight: FontWeight.w900,
                    fontSize: 14.5,
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

/* =========================
   CHIP
========================= */

class _Chip extends StatelessWidget {
  final String text;
  const _Chip({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: AppColors.textDark.withOpacity(0.78),
          fontWeight: FontWeight.w800,
          fontSize: 12.5,
        ),
      ),
    );
  }
}

/* =========================
   BULLET
========================= */

class _Bullet extends StatelessWidget {
  final String text;
  const _Bullet({required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 6),
            decoration: BoxDecoration(
              color: AppColors.darkGreen.withOpacity(0.85),
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.78),
                fontWeight: FontWeight.w700,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/* =========================
   FERTILIZER MODERN CARD
========================= */

class _FertilizerModernCard extends StatelessWidget {
  final List<String> basal;
  final List<String> top;
  final List<String> optional;
  final List<String> tips;

  const _FertilizerModernCard({
    required this.basal,
    required this.top,
    required this.optional,
    required this.tips,
  });

  @override
  Widget build(BuildContext context) {
    Widget block({
      required String title,
      required IconData icon,
      required List<String> rows,
    }) {
      if (rows.isEmpty) return const SizedBox.shrink();

      return Container(
        margin: const EdgeInsets.only(top: 10),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: AppColors.darkGreen, size: 18),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark.withOpacity(0.86),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ...rows.map((e) => _Bullet(text: e)),
          ],
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 34,
                height: 34,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: const Icon(Icons.science_rounded,
                    color: AppColors.darkGreen, size: 18),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  "Fertilizer Recommendation (per hectare)",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 15.5,
                    color: AppColors.textDark,
                  ),
                ),
              ),
            ],
          ),
          block(title: "Basal", icon: Icons.layers_rounded, rows: basal),
          block(
              title: "Top Dressing",
              icon: Icons.trending_up_rounded,
              rows: top),
          block(
              title: "Optional", icon: Icons.extension_rounded, rows: optional),
          block(title: "Tips", icon: Icons.lightbulb_rounded, rows: tips),
        ],
      ),
    );
  }
}

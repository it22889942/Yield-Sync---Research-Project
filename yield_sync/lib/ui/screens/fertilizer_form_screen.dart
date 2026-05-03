import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';

import '../../services/fertiliser_service.dart';
import '../../services/fertiliser_history_service.dart';
import 'fertilizer_result_screen.dart';

class FertilizerFormScreen extends StatefulWidget {
  const FertilizerFormScreen({super.key});

  @override
  State<FertilizerFormScreen> createState() => _FertilizerFormScreenState();
}

class _FertilizerFormScreenState extends State<FertilizerFormScreen> {
  final _formKey = GlobalKey<FormState>();

  final _tempCtrl = TextEditingController(text: "28");
  final _phCtrl = TextEditingController();
  final _nCtrl = TextEditingController();
  final _pCtrl = TextEditingController();
  final _kCtrl = TextEditingController();

  // UI values
  final List<String> _crops = const ["Beetroot", "Red Onion", "Rice", "Radish"];
  final List<String> _stages = const ["Seeding", "Vegetative", "Flowering"];

  String _crop = "Rice";
  String _stage = "Flowering";

  bool _loading = false;

  // interactive temp slider
  double _tempSlider = 28;

  // ✅ header banner asset path
  static const String _bannerAsset =
      "assets/images/paddy-farmers-fertilizer.jpg";

  @override
  void initState() {
    super.initState();
    _tempSlider = double.tryParse(_tempCtrl.text.trim()) ?? 28;
  }

  @override
  void dispose() {
    _tempCtrl.dispose();
    _phCtrl.dispose();
    _nCtrl.dispose();
    _pCtrl.dispose();
    _kCtrl.dispose();
    super.dispose();
  }

  double _toDouble(String s) => double.tryParse(s.trim()) ?? 0.0;

  // backend stage expects lowercase like "flowering"
  String _toBackendStage(String ui) => ui.toLowerCase();

  Future<void> _analyze() async {
    if (!_formKey.currentState!.validate()) return;

    final temp = _toDouble(_tempCtrl.text);
    final ph = _toDouble(_phCtrl.text);
    final n = _toDouble(_nCtrl.text);
    final p = _toDouble(_pCtrl.text);
    final k = _toDouble(_kCtrl.text);

    final cropBackend = _crop;
    final stageBackend = _toBackendStage(_stage);

    setState(() => _loading = true);

    try {
      final res = await FertiliserService.predict(
        temperature: temp,
        ph: ph,
        nitrogen: n,
        phosphorous: p,
        potassium: k,
        crop: cropBackend,
        growthStage: stageBackend,
      );

      final request = {
        "temperature": temp,
        "ph": ph,
        "nitrogen": n,
        "phosphorous": p,
        "potassium": k,
        "crop": cropBackend,
        "growth_stage": stageBackend,
      };

      await FertiliserHistoryService().saveFertiliserPrediction(
        request: request,
        response: res.toJson(),
      );

      if (!mounted) return;

      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => FertilizerResultScreen(
            result: res,
            cropLabel: _crop,
            stageLabel: _stage,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Analyze error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _reset() {
    setState(() {
      _tempCtrl.text = "28";
      _tempSlider = 28;
      _phCtrl.clear();
      _nCtrl.clear();
      _pCtrl.clear();
      _kCtrl.clear();
      _crop = "Rice";
      _stage = "Flowering";
    });
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===================== HERO HEADER =====================
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
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.arrow_back_ios_new_rounded),
                      color: Colors.white,
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        "Fertilizer Recommendation",
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.92),
                          fontWeight: FontWeight.w900,
                          fontSize: 16,
                        ),
                      ),
                    ),
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
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Smart Nutrient Plan",
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.96),
                      fontWeight: FontWeight.w900,
                      fontSize: 22,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Enter field + crop details to get the best mix",
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.74),
                      fontWeight: FontWeight.w700,
                      fontSize: 12.8,
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                Row(
                  children: [
                    Expanded(
                      child: _headerChip(
                        icon: Icons.spa_rounded,
                        label: "Crop: $_crop",
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _headerChip(
                        icon: Icons.timeline_rounded,
                        label: "Stage: $_stage",
                      ),
                    ),
                  ],
                ),

                // ✅ Banner image
                const SizedBox(height: 12),
                Container(
                  height: 110,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: Colors.white.withOpacity(0.14)),
                    color: Colors.white.withOpacity(0.08),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.asset(_bannerAsset, fit: BoxFit.cover),
                        Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                Colors.black.withOpacity(0.05),
                                Colors.black.withOpacity(0.55),
                              ],
                            ),
                          ),
                        ),
                        Align(
                          alignment: Alignment.bottomLeft,
                          child: Padding(
                            padding: const EdgeInsets.all(10),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.18),
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.25),
                                ),
                              ),
                              child: Text(
                                "Crop Guide",
                                style: TextStyle(
                                  color: Colors.white.withOpacity(0.95),
                                  fontWeight: FontWeight.w900,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===================== FORM BODY =====================
          Expanded(
            child: Form(
              key: _formKey,
              child: ListView(
                // ✅ bottom padding reduced because button is inside list now
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
                children: [
                  _infoCard(
                    title:
                        "Tip: Use your soil test values for accurate results.",
                    leftIcon: Icons.tips_and_updates_rounded,
                    actionText: "Reset",
                    onAction: _reset,
                  ),
                  const SizedBox(height: 12),

                  _section(
                    icon: Icons.cloud_rounded,
                    title: "Environment",
                    subtitle: "Temperature + soil pH",
                    child: Column(
                      children: [
                        _card(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  const Icon(Icons.thermostat_rounded,
                                      color: AppColors.darkGreen),
                                  const SizedBox(width: 8),
                                  const Text(
                                    "Temperature",
                                    style: TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color: AppColors.textDark,
                                    ),
                                  ),
                                  const Spacer(),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 10, vertical: 6),
                                    decoration: BoxDecoration(
                                      color:
                                          AppColors.primary.withOpacity(0.14),
                                      borderRadius: BorderRadius.circular(999),
                                      border:
                                          Border.all(color: AppColors.border),
                                    ),
                                    child: Text(
                                      "${_tempSlider.toStringAsFixed(0)}°C",
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w900,
                                        color: AppColors.darkGreen,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              Slider(
                                value: _tempSlider.clamp(0, 45),
                                min: 0,
                                max: 45,
                                divisions: 45,
                                onChanged: (v) {
                                  setState(() {
                                    _tempSlider = v;
                                    _tempCtrl.text = v.toStringAsFixed(0);
                                  });
                                },
                              ),
                              const SizedBox(height: 6),
                              _input(
                                label: "Temperature (°C)",
                                ctrl: _tempCtrl,
                                suffix: "°C",
                                keyboard: TextInputType.number,
                                validator: (v) => _reqNumber(v, "Temperature"),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 10),
                        _card(
                          child: Column(
                            children: [
                              _input(
                                label: "Soil pH",
                                ctrl: _phCtrl,
                                suffix: "pH",
                                keyboard: TextInputType.number,
                                validator: (v) => _reqNumber(v, "pH"),
                              ),
                              const SizedBox(height: 8),
                              Align(
                                alignment: Alignment.centerLeft,
                                child: Text(
                                  "Typical range: 5.5 – 7.5",
                                  style: TextStyle(
                                    color: AppColors.textDark.withOpacity(0.55),
                                    fontWeight: FontWeight.w700,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 12),

                  _section(
                    icon: Icons.science_rounded,
                    title: "Soil Nutrients",
                    subtitle: "N • P • K values",
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: _input(
                                label: "Nitrogen (N)",
                                ctrl: _nCtrl,
                                suffix: "N",
                                keyboard: TextInputType.number,
                                validator: (v) => _reqNumber(v, "Nitrogen"),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: _input(
                                label: "Phosphorous (P)",
                                ctrl: _pCtrl,
                                suffix: "P",
                                keyboard: TextInputType.number,
                                validator: (v) => _reqNumber(v, "Phosphorous"),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        _input(
                          label: "Potassium (K)",
                          ctrl: _kCtrl,
                          suffix: "K",
                          keyboard: TextInputType.number,
                          validator: (v) => _reqNumber(v, "Potassium"),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 12),

                  _section(
                    icon: Icons.spa_rounded,
                    title: "Crop Details",
                    subtitle: "Crop + growth stage",
                    child: Column(
                      children: [
                        _dropdown(
                          label: "Crop",
                          value: _crop,
                          items: _crops,
                          onChanged: (v) => setState(() => _crop = v!),
                        ),
                        const SizedBox(height: 10),
                        _dropdown(
                          label: "Growth Stage",
                          value: _stage,
                          items: _stages,
                          onChanged: (v) => setState(() => _stage = v!),
                        ),
                      ],
                    ),
                  ),

                  // ✅ Analyze button moved into scroll (NOT sticky)
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton.icon(
                      onPressed: _loading ? null : _analyze,
                      icon: _loading
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: AppColors.darkGreen,
                              ),
                            )
                          : const Icon(Icons.auto_graph_rounded),
                      label: Text(
                        _loading ? "Analyzing..." : "Analyze",
                        style: const TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 16,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: AppColors.darkGreen,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 18),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===================== UI HELPERS =====================

  Widget _headerChip({required IconData icon, required String label}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.16)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: Colors.white.withOpacity(0.95)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withOpacity(0.92),
                fontWeight: FontWeight.w800,
                fontSize: 12.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _section({
    required IconData icon,
    required String title,
    required String subtitle,
    required Widget child,
  }) {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(icon, color: AppColors.darkGreen, size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        color: AppColors.textDark,
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: AppColors.textDark.withOpacity(0.55),
                        fontSize: 12.3,
                      ),
                    ),
                  ],
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

  Widget _infoCard({
    required String title,
    required IconData leftIcon,
    String? actionText,
    VoidCallback? onAction,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
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
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.18),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(leftIcon, color: AppColors.darkGreen),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              title,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.85),
                fontWeight: FontWeight.w700,
                height: 1.25,
              ),
            ),
          ),
          if (actionText != null && onAction != null) ...[
            const SizedBox(width: 10),
            TextButton(
              onPressed: onAction,
              child: Text(
                actionText,
                style: const TextStyle(fontWeight: FontWeight.w900),
              ),
            )
          ]
        ],
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _input({
    required String label,
    required TextEditingController ctrl,
    required String suffix,
    required TextInputType keyboard,
    required String? Function(String?) validator,
  }) {
    return TextFormField(
      controller: ctrl,
      keyboardType: keyboard,
      validator: validator,
      decoration: InputDecoration(
        labelText: label,
        suffixText: suffix,
        filled: true,
        fillColor: AppColors.surface,
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
          borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
        ),
      ),
    );
  }

  Widget _dropdown({
    required String label,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      onChanged: onChanged,
      decoration: InputDecoration(
        labelText: label,
        filled: true,
        fillColor: AppColors.surface,
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
          borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
        ),
      ),
      items: items
          .map((e) => DropdownMenuItem<String>(value: e, child: Text(e)))
          .toList(),
    );
  }

  String? _reqNumber(String? v, String name) {
    if (v == null || v.trim().isEmpty) return "$name is required";
    if (double.tryParse(v.trim()) == null) return "Enter a valid number";
    return null;
  }
}

import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../utils/form_validators.dart';
import '../../services/labour_equipment_post_service.dart';
import '../../services/nav.dart';
import '../widgets/app_text_field.dart';
import '../widgets/post_availability_schedule.dart';
import '../widgets/primary_button.dart';

class AddEquipmentPostScreen extends StatefulWidget {
  const AddEquipmentPostScreen({super.key});

  @override
  State<AddEquipmentPostScreen> createState() => _AddEquipmentPostScreenState();
}

class _AddEquipmentPostScreenState extends State<AddEquipmentPostScreen> {
  final _formKey = GlobalKey<FormState>();
  final _hourlyRateCtrl = TextEditingController();
  final _dailyRateCtrl = TextEditingController();
  final _ownerNameCtrl = TextEditingController();
  final _ownerContactCtrl = TextEditingController();

  bool _saving = false;

  String? _selectedEquipmentType;
  String? _selectedCropType;
  String? _selectedDistrict;
  String? _selectedSeason;
  String? _selectedCondition;
  String? _selectedAvailability;
  TimeOfDay? _availableFrom;
  TimeOfDay? _availableTo;

  static const List<String> _equipmentTypes = [
    'Tractor',
    'Combine Harvester',
    'Harvester',
    'Rotavator',
    'Plough',
    'Power Sprayer',
    'Seed Drill',
    'Trailer',
    'Water Pump',
    'Transplanter',
    'Grass Cutter',
    'Fertilizer Spreader',
    'Other',
  ];

  static const List<String> _districts = [
    'Alawwa',
    'Bingiriya',
    'Galgamuwa',
    'Giriulla',
    'Hettipola',
    'Hiriyala',
    'Ibbagamuwa',
    'Kuliyapitiya',
    'Kurunegala',
    'Maho',
    'Maspotha',
    'Mawathagama',
    'Narammala',
    'Nikaweratiya',
    'Pannala',
    'Polgahawela',
    'Rideegama',
    'Wariyapola',
    'Weerambugedara',
  ];

  static const List<String> _seasons = ['Yala', 'Maha'];

  static const List<String> _cropTypes = [
    'Paddy',
    'Maize',
    'Onion',
    'Chickpea',
    'Vegetables',
    'Other',
  ];

  static const List<String> _conditions = [
    'Excellent',
    'Good',
    'Fair',
    'Needs service',
  ];

  static const List<String> _availabilityOptions = [
    'Weekdays',
    'Weekends',
    'Both',
  ];

  @override
  void dispose() {
    _hourlyRateCtrl.dispose();
    _dailyRateCtrl.dispose();
    _ownerNameCtrl.dispose();
    _ownerContactCtrl.dispose();
    super.dispose();
  }

  String _formatTimeOfDay(TimeOfDay time) {
    final hour = time.hourOfPeriod == 0 ? 12 : time.hourOfPeriod;
    final minute = time.minute.toString().padLeft(2, '0');
    final suffix = time.period == DayPeriod.am ? 'AM' : 'PM';
    return '$hour:$minute $suffix';
  }

  String _formatLocation(String city) {
    if (city.toLowerCase() == 'kurunegala') return 'Kurunegala';
    return 'Kurunegala, $city';
  }

  Future<void> _pickFromTime() async {
    final selected = await showAppTimePicker(
      context,
      initialTime: _availableFrom ?? const TimeOfDay(hour: 8, minute: 0),
      helpText: 'Start time',
    );
    if (selected == null) return;
    setState(() => _availableFrom = selected);
  }

  Future<void> _pickToTime() async {
    final selected = await showAppTimePicker(
      context,
      initialTime: _availableTo ?? const TimeOfDay(hour: 17, minute: 0),
      helpText: 'End time',
    );
    if (selected == null) return;
    setState(() => _availableTo = selected);
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_availableFrom == null || _availableTo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Select available time range')),
      );
      return;
    }
    final fromMinutes = _availableFrom!.hour * 60 + _availableFrom!.minute;
    final toMinutes = _availableTo!.hour * 60 + _availableTo!.minute;
    if (toMinutes <= fromMinutes) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            '"Available to" must be later than "Available from"',
          ),
        ),
      );
      return;
    }

    setState(() => _saving = true);
    try {
      final hourly = double.tryParse(_hourlyRateCtrl.text.trim()) ?? 0.0;
      final daily = double.tryParse(_dailyRateCtrl.text.trim()) ?? 0.0;
      if (hourly <= 0 && daily <= 0) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                'Enter at least one rate: hourly or daily (LKR)',
              ),
            ),
          );
        }
        setState(() => _saving = false);
        return;
      }

      final availableTime =
          '${_formatTimeOfDay(_availableFrom!)} - ${_formatTimeOfDay(_availableTo!)}';

      await LabourEquipmentPostService.createEquipmentPost(
        equipmentType: _selectedEquipmentType!,
        forCrop: _selectedCropType!,
        nearestMajorDistrict: _formatLocation(_selectedDistrict!),
        season: _selectedSeason!,
        condition: _selectedCondition!,
        availableDay: _selectedAvailability!,
        availableTime: availableTime,
        hourlyRateLkr: hourly > 0 ? hourly : (daily / 8),
        dailyRateLkr: daily > 0 ? daily : (hourly * 8),
        ownerName: _ownerNameCtrl.text.trim(),
        ownerContact: _ownerContactCtrl.text.trim(),
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Equipment listed. It will appear in the list.'),
        ),
      );
      Nav.back(context);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            _header(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _card(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Equipment details',
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                fontSize: 16,
                                color: AppColors.textDark,
                              ),
                            ),
                            const SizedBox(height: 14),
                            _dropdownField(
                              label: 'Equipment type',
                              value: _selectedEquipmentType,
                              items: _equipmentTypes,
                              onChanged: (value) =>
                                  setState(() => _selectedEquipmentType = value),
                              validatorMessage: 'Equipment type required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'For crop',
                              value: _selectedCropType,
                              items: _cropTypes,
                              onChanged: (value) =>
                                  setState(() => _selectedCropType = value),
                              validatorMessage: 'Crop type required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Location (Kurunegala)',
                              value: _selectedDistrict,
                              items: _districts,
                              onChanged: (value) =>
                                  setState(() => _selectedDistrict = value),
                              validatorMessage: 'Location required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Season',
                              value: _selectedSeason,
                              items: _seasons,
                              onChanged: (value) =>
                                  setState(() => _selectedSeason = value),
                              validatorMessage: 'Season required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Condition',
                              value: _selectedCondition,
                              items: _conditions,
                              onChanged: (value) =>
                                  setState(() => _selectedCondition = value),
                              validatorMessage: 'Condition required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Available days',
                              value: _selectedAvailability,
                              items: _availabilityOptions,
                              onChanged: (value) =>
                                  setState(() => _selectedAvailability = value),
                              validatorMessage: 'Availability required',
                            ),
                            const SizedBox(height: 12),
                            PostAvailabilityTimeSection(
                              fromDisplay: _availableFrom == null
                                  ? null
                                  : _formatTimeOfDay(_availableFrom!),
                              toDisplay: _availableTo == null
                                  ? null
                                  : _formatTimeOfDay(_availableTo!),
                              onPickFrom: _pickFromTime,
                              onPickTo: _pickToTime,
                              onClearFrom: () =>
                                  setState(() => _availableFrom = null),
                              onClearTo: () =>
                                  setState(() => _availableTo = null),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _hourlyRateCtrl,
                              label: 'Hourly rate (LKR)',
                              hint: 'e.g. 2000',
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                decimal: true,
                              ),
                              validator: (v) =>
                                  FormValidators.optionalPositiveNumber(
                                v,
                                'Hourly rate (LKR)',
                              ),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _dailyRateCtrl,
                              label: 'Daily rate (LKR)',
                              hint: 'e.g. 15000',
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                decimal: true,
                              ),
                              validator: (v) =>
                                  FormValidators.optionalPositiveNumber(
                                v,
                                'Daily rate (LKR)',
                              ),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _ownerNameCtrl,
                              label: 'Owner name',
                              hint:
                                  'Your name (or leave blank to use profile)',
                              textCapitalization: TextCapitalization.words,
                              validator: (v) =>
                                  FormValidators.optionalPersonName(
                                v,
                                fieldLabel: 'Owner name',
                              ),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _ownerContactCtrl,
                              label: 'Contact number',
                              hint: '07XXXXXXXX (10 digits)',
                              keyboardType: TextInputType.phone,
                              validator: FormValidators.optionalPhone10Digits,
                            ),
                            const SizedBox(height: 18),
                            PrimaryButton(
                              text: _saving ? 'Adding...' : 'List equipment',
                              icon: _saving
                                  ? Icons.hourglass_top_rounded
                                  : Icons.add_business_rounded,
                              onPressed: _saving ? null : _submit,
                              isLoading: _saving,
                            ),
                          ],
                        ),
                      ),
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

  Widget _header() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Nav.back(context),
            icon: const Icon(Icons.arrow_back_ios_new_rounded),
            color: Colors.white,
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'List equipment for hire',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w900,
                fontSize: 18,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _dropdownField({
    required String label,
    required String? value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
    required String validatorMessage,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      isExpanded: true,
      icon: const Icon(
        Icons.keyboard_arrow_down_rounded,
        color: AppColors.darkGreen,
      ),
      borderRadius: BorderRadius.circular(14),
      dropdownColor: Colors.white,
      style: const TextStyle(
        color: AppColors.textDark,
        fontWeight: FontWeight.w600,
      ),
      decoration: _inputDecoration(label: label),
      hint: const Text(
        'Tap to open list',
        style: TextStyle(
          color: AppColors.muted,
          fontWeight: FontWeight.w600,
        ),
      ),
      items: items
          .map(
            (item) => DropdownMenuItem<String>(
              value: item,
              child: Text(
                item,
                style: const TextStyle(
                  color: AppColors.textDark,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          )
          .toList(),
      onChanged: onChanged,
      validator: (v) => v == null || v.isEmpty ? validatorMessage : null,
    );
  }

  InputDecoration _inputDecoration({
    required String label,
    Widget? suffixIcon,
  }) {
    const borderRadius = BorderRadius.all(Radius.circular(14));
    return InputDecoration(
      labelText: label,
      suffixIcon: suffixIcon,
      filled: true,
      fillColor: const Color(0xFFF7FCFA),
      contentPadding:
          const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      labelStyle: const TextStyle(
        color: AppColors.muted,
        fontWeight: FontWeight.w600,
      ),
      enabledBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.border),
      ),
      focusedBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.primary, width: 1.6),
      ),
      errorBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.error),
      ),
      focusedErrorBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.error, width: 1.4),
      ),
    );
  }
}

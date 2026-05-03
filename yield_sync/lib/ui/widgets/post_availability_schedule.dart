import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

/// Themed clock-dial time picker with visible Done/Cancel and app colors.
Future<TimeOfDay?> showAppTimePicker(
  BuildContext context, {
  required TimeOfDay initialTime,
  required String helpText,
}) {
  return showTimePicker(
    context: context,
    initialTime: initialTime,
    initialEntryMode: TimePickerEntryMode.dial,
    helpText: helpText,
    cancelText: 'Cancel',
    confirmText: 'Done',
    builder: (context, child) {
      final base = Theme.of(context);
      return Theme(
        data: base.copyWith(
          colorScheme: base.colorScheme.copyWith(
            primary: AppColors.darkGreen,
            onPrimary: Colors.white,
            surface: Colors.white,
          ),
          textButtonTheme: TextButtonThemeData(
            style: TextButton.styleFrom(
              foregroundColor: AppColors.darkGreen,
              textStyle: const TextStyle(fontWeight: FontWeight.w800),
            ),
          ),
          timePickerTheme: TimePickerThemeData(
            backgroundColor: Colors.white,
            dialHandColor: AppColors.darkGreen,
            dialBackgroundColor: AppColors.surface,
            hourMinuteTextColor: WidgetStateColor.resolveWith((states) {
              if (states.contains(WidgetState.selected)) {
                return Colors.white;
              }
              return AppColors.textDark;
            }),
            dayPeriodColor: WidgetStateColor.resolveWith((states) {
              if (states.contains(WidgetState.selected)) {
                return AppColors.darkGreen;
              }
              return AppColors.surface;
            }),
            dayPeriodTextColor: WidgetStateColor.resolveWith((states) {
              if (states.contains(WidgetState.selected)) {
                return Colors.white;
              }
              return AppColors.textDark;
            }),
            entryModeIconColor: AppColors.darkGreen,
            helpTextStyle: const TextStyle(
              color: AppColors.textDark,
              fontWeight: FontWeight.w800,
              fontSize: 17,
            ),
          ),
        ),
        child: MediaQuery(
          data: MediaQuery.of(context).copyWith(alwaysUse24HourFormat: false),
          child: child!,
        ),
      );
    },
  );
}

/// Large tap targets + helper copy + per-field Clear for labour/equipment post flows.
class PostAvailabilityTimeSection extends StatelessWidget {
  final String? fromDisplay;
  final String? toDisplay;
  final VoidCallback onPickFrom;
  final VoidCallback onPickTo;
  final VoidCallback onClearFrom;
  final VoidCallback onClearTo;

  const PostAvailabilityTimeSection({
    super.key,
    required this.fromDisplay,
    required this.toDisplay,
    required this.onPickFrom,
    required this.onPickTo,
    required this.onClearFrom,
    required this.onClearTo,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'Working hours',
          style: TextStyle(
            fontWeight: FontWeight.w900,
            fontSize: 14,
            color: AppColors.textDark,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Tap a card to open the clock. Use Clear to remove a time and pick again.',
          style: TextStyle(
            fontSize: 12.5,
            height: 1.35,
            color: AppColors.muted,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 12),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: _TimeChoiceCard(
                title: 'Available from',
                value: fromDisplay,
                onTap: onPickFrom,
                onClear: fromDisplay == null ? null : onClearFrom,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _TimeChoiceCard(
                title: 'Available to',
                value: toDisplay,
                onTap: onPickTo,
                onClear: toDisplay == null ? null : onClearTo,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _TimeChoiceCard extends StatelessWidget {
  final String title;
  final String? value;
  final VoidCallback onTap;
  final VoidCallback? onClear;

  const _TimeChoiceCard({
    required this.title,
    required this.value,
    required this.onTap,
    this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    final hasValue = value != null && value!.isNotEmpty;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Ink(
          decoration: BoxDecoration(
            color: const Color(0xFFF7FCFA),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: hasValue ? AppColors.primary : AppColors.border,
              width: hasValue ? 2 : 1.2,
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 6, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text(
                        title,
                        style: const TextStyle(
                          fontWeight: FontWeight.w800,
                          fontSize: 12.5,
                          color: AppColors.muted,
                        ),
                      ),
                    ),
                    if (hasValue && onClear != null)
                      IconButton(
                        tooltip: 'Clear time',
                        visualDensity: VisualDensity.compact,
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(
                          minWidth: 40,
                          minHeight: 40,
                        ),
                        onPressed: onClear,
                        icon: const Icon(
                          Icons.close_rounded,
                          size: 22,
                          color: AppColors.muted,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    Icon(
                      Icons.schedule_rounded,
                      size: 28,
                      color: hasValue ? AppColors.darkGreen : AppColors.muted,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        hasValue ? value! : 'Tap to choose time',
                        style: TextStyle(
                          fontSize: hasValue ? 19 : 14,
                          fontWeight: FontWeight.w800,
                          letterSpacing: hasValue ? 0.3 : 0,
                          color:
                              hasValue ? AppColors.textDark : AppColors.muted,
                          height: 1.2,
                        ),
                      ),
                    ),
                  ],
                ),
                if (hasValue) ...[
                  const SizedBox(height: 8),
                  const Text(
                    'Tap card again to change',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: AppColors.muted,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

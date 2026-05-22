import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:permission_handler/permission_handler.dart';

const int IMG_SIZE = 75;
const String MODEL_PATH = 'assets/emotion_model_with_new_dataset.tflite';
const String LABELS_PATH = 'assets/labels.txt';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const EmotionDetectorScreen(),
    );
  }
}

class EmotionDetectorScreen extends StatefulWidget {
  const EmotionDetectorScreen({super.key});

  @override
  State<EmotionDetectorScreen> createState() => _EmotionDetectorScreenState();
}

class _EmotionDetectorScreenState extends State<EmotionDetectorScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<CameraDescription> _cameras = [];

  bool _isBusy = false;
  bool _isSwitching = false;

  String _currentEmotion = "Memuat...";
  double _confidence = 0.0;

  int _selectedCameraIndex = 0;

  final List<String> _predictionHistory = [];
  final int _historyLimit = 10;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initAll();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _controller;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _setupCamera();
    }
  }

  Future<void> _initAll() async {
    await _requestPermission();
    await _loadModel();

    try {
      _cameras = await availableCameras();
      if (_cameras.isNotEmpty) {
        int frontIndex = _cameras.indexWhere(
          (c) => c.lensDirection == CameraLensDirection.front,
        );
        _selectedCameraIndex = (frontIndex != -1) ? frontIndex : 0;
        await _setupCamera();
      }
    } catch (e) {
      print("❌ Error camera list: $e");
    }
  }

  Future<void> _requestPermission() async {
    await Permission.camera.request();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(MODEL_PATH);
      final labelData = await rootBundle.loadString(LABELS_PATH);
      _labels = labelData
          .split('\n')
          .where((s) => s.trim().isNotEmpty)
          .toList();
      print("✅ Model Loaded (${_labels.length} classes)");
    } catch (e) {
      print("❌ Error Load Model: $e");
    }
  }

  Future<void> _switchCamera() async {
    if (_isSwitching || _cameras.length < 2) return;

    setState(() {
      _isSwitching = true;
    });

    try {
      if (_controller != null) {
        if (_controller!.value.isStreamingImages) {
          await _controller!.stopImageStream();
        }
        await _controller!.dispose();
        _controller = null;
      }

      await Future.delayed(const Duration(milliseconds: 300));

      int newIndex = (_selectedCameraIndex + 1) % _cameras.length;
      _selectedCameraIndex = newIndex;
      print("🔄 Switch ke Index: $_selectedCameraIndex");

      await _setupCamera();
    } catch (e) {
      print("❌ Error Switch Camera: $e");
    } finally {
      if (mounted) {
        setState(() {
          _isSwitching = false;
        });
      }
    }
  }

  Future<void> _setupCamera() async {
    if (_cameras.isEmpty) return;

    final camera = _cameras[_selectedCameraIndex];

    _controller = CameraController(
      camera,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;

      _controller!.startImageStream((image) {
        if (!_isBusy && !_isSwitching) {
          _isBusy = true;
          _processFrame(image);
        }
      });

      setState(() {});
    } catch (e) {
      print("❌ Error Setup Kamera: $e");
    }
  }

  Future<void> _processFrame(CameraImage cameraImage) async {
    if (_interpreter == null || _labels.isEmpty) {
      _isBusy = false;
      return;
    }

    try {
      await Future.delayed(Duration.zero);
      if (!mounted) return;

      img.Image? image = _convertCameraImage(cameraImage);
      if (image == null) return;

      img.Image resized = img.copyResize(
        image,
        width: IMG_SIZE,
        height: IMG_SIZE,
      );

      var input = List.generate(
        1,
        (i) => List.generate(
          IMG_SIZE,
          (y) => List.generate(IMG_SIZE, (x) {
            var pixel = resized.getPixel(x, y);
            return [pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble()];
          }),
        ),
      );

      // final input = Float32List(1 * IMG_SIZE * IMG_SIZE);
      // int idx = 0;

      // for (int y = 0; y < IMG_SIZE; y++) {
      //   for (int x = 0; x < IMG_SIZE; x++) {
      //     final p = resized.getPixel(x, y);
      //     input[idx++] = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b) / 255.0;
      //   }
      // }

      // final inputTensor = input.reshape([1, IMG_SIZE, IMG_SIZE, 1]);

      var output = List.filled(
        1 * _labels.length,
        0.0,
      ).reshape([1, _labels.length]);
      _interpreter!.run(input, output);

      List<double> probabilities = List<double>.from(output[0]);
      int maxIndex = 0;
      double maxProb = 0.0;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      if (mounted) {
        _applySmoothing(_labels[maxIndex], maxProb);
      }
    } catch (e) {
      print("⚠️ Error Processing: $e");
    } finally {
      _isBusy = false;
    }
  }

  void _applySmoothing(String newEmotion, double newConf) {
    _predictionHistory.add(newEmotion);
    if (_predictionHistory.length > _historyLimit)
      _predictionHistory.removeAt(0);

    Map<String, int> frequency = {};
    for (var emotion in _predictionHistory) {
      frequency[emotion] = (frequency[emotion] ?? 0) + 1;
    }

    String smoothEmotion = frequency.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;

    if (mounted) {
      setState(() {
        _currentEmotion = smoothEmotion.toUpperCase();
        _confidence = newConf;
      });
    }
  }

  img.Image? _convertCameraImage(CameraImage image) {
    try {
      if (image.format.group == ImageFormatGroup.yuv420) {
        return _convertYUV420(image);
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        return _convertBGRA8888(image);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  img.Image _convertYUV420(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    final img.Image out = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            (uvPixelStride * (x / 2).floor()) + (uvRowStride * (y / 2).floor());
        final int index = y * width + x;

        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        int r = (yp + (vp * 1.402 - 179.456)).toInt().clamp(0, 255);
        int g = (yp - (up * 0.34414 - 44.814) - (vp * 0.71414 - 99.135))
            .toInt()
            .clamp(0, 255);
        int b = (yp + (up * 1.772 - 226.816)).toInt().clamp(0, 255);

        out.setPixelRgb(x, y, r, g, b);
      }
    }
    return img.copyRotate(out, angle: 90);
  }

  img.Image _convertBGRA8888(CameraImage image) {
    return img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes[0].bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }

  @override
  Widget build(BuildContext context) {
    bool isInitializing =
        _controller == null || !_controller!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          if (isInitializing)
            const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 10),
                  Text(
                    "Menyiapkan Kamera...",
                    style: TextStyle(color: Colors.white),
                  ),
                ],
              ),
            )
          else
            Positioned.fill(child: CameraPreview(_controller!)),

          if (!isInitializing)
            Positioned(
              bottom: 50,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: _getEmotionColor(_currentEmotion).withOpacity(0.8),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  children: [
                    Text(
                      _currentEmotion,
                      style: const TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      "Confidence: ${(_confidence * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(color: Colors.white70),
                    ),
                  ],
                ),
              ),
            ),

          Positioned(
            top: 50,
            right: 20,
            child: FloatingActionButton(
              heroTag: "cam_switch",
              backgroundColor: _isSwitching ? Colors.grey : Colors.black54,
              onPressed: _isSwitching ? null : _switchCamera,
              child: _isSwitching
                  ? const Padding(
                      padding: EdgeInsets.all(10),
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Icon(Icons.cameraswitch, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }

  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'bahagia':
        return Colors.green;
      case 'marah':
        return Colors.red;
      case 'sedih':
        return Colors.blue;
      case 'takut':
        return Colors.purple;
      case 'jijik':
        return Colors.brown;
      case 'netral':
        return Colors.grey;
      default:
        return Colors.orange;
    }
  }
}

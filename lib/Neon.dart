import 'package:flutter/material.dart';

//void main() {
//  runApp(const MyApp());
//}

class MyApp1 extends StatelessWidget {
  const MyApp1({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool isPressed = false;

  @override
  Widget build(BuildContext context) {
    Color shadowColor = Colors.red;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            boxShadow: [
              for (double i = 1; i < (isPressed ? 8 : 4); i++)
                BoxShadow(
                    color: shadowColor,
                    blurRadius: i * 3,
                    blurStyle: BlurStyle.outer,
                    spreadRadius: -1),
            ],
          ),
          child: TextButton(
            onHover: (hovered) {
              setState(() {
                isPressed = hovered;
              });
            },
            style: TextButton.styleFrom(
              side: const BorderSide(
                color: Colors.white,
                width: 4,
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            onPressed: () {},
            child: Listener(
              onPointerDown: (_) {
                setState(() {
                  isPressed = true;
                });
              },
              onPointerUp: (_) {
                setState(() {
                  isPressed = false;
                });
              },
              child: Text(
                "Neon Button",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 40,
                  shadows: [
                    for (double i = 1; i < (isPressed ? 8 : 4); i++)
                      Shadow(color: shadowColor, blurRadius: i * 3),
                    // Shadow(color: shadowColor, blurRadius: 6),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:image_picker/image_picker.dart';

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  bool _loading = true;
  late File _image;
  late List Data;
  var _output;
  final picker = ImagePicker(); //allows us to pick image from gallery or camera

  @override
  void initState() {
    //initS is the first function that is executed by default when this class is called
    super.initState();
    loadModel().then((value) {
      setState(() {});
    });
  }

  @override
  void dispose() {
    //dis function disposes and clears our memory
    super.dispose();
    Tflite.close();
  }

  //このメソッド→すると、画像に対して分類モデルが実行されます。
  //numResultsは、私たちが持っているクラスの数であり、変更を保存するためにsetStateを追加します。
  classifyImage(List Data) async {
    //this function runs the model on the image
    var output = await Tflite.runModelOnBinary(
      binary: binary,
      numResults: 5, //the amout of categories our neural network can predict
      threshold: 0.5,
      //imageMean: 127.5,
      //imageStd: 127.5,
      asynch: true,
    );
    setState(() {
      _output = output;
      _loading = false;
    });
  }

  //この関数はモデルをロードするので、initSメソッドの中に入れます。
  loadModel() async {
    //this function loads our model
    await Tflite.loadModel(
        model: 'assets/nk225_model.tflite', labels: 'assets/labels.txt');
  }

  //この機能は、カメラから画像を取得するために使用されます。
  pickImage() async {
    //this function to grab the image from camera
    var image = await picker.getImage(source: ImageSource.camera);
    if (image == null) return null;

    setState(() {
      _image = File(image.path);
    });
    classifyImage(Data); //このメソッド→すると、画像に対して分類モデルが実行されます。
  }

  //この機能は、ユーザーのギャラリーから画像を取得するために使用されます。
  pickGalleryImage() async {
    //this function to grab the image from gallery
    var image = await picker.getImage(source: ImageSource.gallery);
    if (image == null) return null;

    setState(() {
      _image = File(image.path);
    });
    classifyImage(_image);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: Text(
          'Fruits and Veggies Neural Network',
          style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w200,
              fontSize: 20,
              letterSpacing: 0.8),
        ),
      ),
      body: Container(
        color: Colors.black.withOpacity(0.9),
        padding: EdgeInsets.symmetric(horizontal: 35, vertical: 50),
        child: Container(
          alignment: Alignment.center,
          padding: EdgeInsets.all(30),
          decoration: BoxDecoration(
            color: Color(0xFF2A363B),
            borderRadius: BorderRadius.circular(30),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                child: Center(
                  child: _loading == true
                      ? null //show nothing if no picture selected
                      : Container(
                          child: Column(
                            children: [
                              Container(
                                height: 250,
                                width: 250,
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(30),
                                  child: Image.file(
                                    _image,
                                    fit: BoxFit.fill,
                                  ),
                                ),
                              ),
                              Divider(
                                height: 25,
                                thickness: 1,
                              ),
                              _output != null
                                  ? Text(
                                      'The object is: ${_output[0]['label']}!',
                                      style: TextStyle(
                                          color: Colors.white,
                                          fontSize: 18,
                                          fontWeight: FontWeight.w400),
                                    )
                                  : Container(),
                              Divider(
                                height: 25,
                                thickness: 1,
                              ),
                            ],
                          ),
                        ),
                ),
              ),
              Container(
                child: Column(
                  children: [
                    GestureDetector(
                      onTap: pickImage,
                      child: Container(
                        width: MediaQuery.of(context).size.width - 200,
                        alignment: Alignment.center,
                        padding:
                            EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                        decoration: BoxDecoration(
                            color: Colors.blueGrey[600],
                            borderRadius: BorderRadius.circular(15)),
                        child: Text(
                          'Take A Photo',
                          style: TextStyle(color: Colors.white, fontSize: 16),
                        ),
                      ),
                    ),
                    SizedBox(
                      height: 30,
                    ),
                    GestureDetector(
                      onTap: pickGalleryImage,
                      child: Container(
                        width: MediaQuery.of(context).size.width - 200,
                        alignment: Alignment.center,
                        padding:
                            EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                        decoration: BoxDecoration(
                            color: Colors.blueGrey[600],
                            borderRadius: BorderRadius.circular(15)),
                        child: Text(
                          'Pick From Gallery',
                          style: TextStyle(color: Colors.white, fontSize: 16),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

<template>
  <div style="padding: 20px; max-width: 600px; margin: 0 auto; text-align: center;">
    <h2 style="margin-bottom: 20px;">Melanoma Binary Segmentation</h2>
    
    <div style="margin-bottom: 20px; text-align: center;">
      <label style="display: inline-block;">
        Select BMP Image:
        <input 
          type="file" 
          accept=".bmp"
          @change="handleFileUpload"
          style="display: block; margin: 5px auto;"
        >
      </label>
    </div>
    
    <div style="margin-bottom: 20px; text-align: center;">
      <label style="display: inline-block;">
        Neural Network Type:
        <select 
          v-model="nnType" 
          style="display: block; margin: 5px auto; padding: 5px; width: 200px;"
        >
          <option value="unet">UNET</option>
          <option value="segnet">SegNet</option>
          <option value="naive">Naive</option>
        </select>
      </label>
    </div>

    <div v-if="error" style="color: red; margin-bottom: 10px;">
      {{ error }}
    </div>

    <div v-if="loading" style="margin-bottom: 10px;">
      Processing image...
    </div>

    <div v-if="resultImage" style="text-align: center;">
      <h3>Segmentation Result:</h3>
      <img 
        :src="resultImage" 
        alt="Segmentation result" 
        style="max-width: 100%; margin: 10px auto; display: block;"
      >
      <a 
        :href="resultImage" 
        download="segmentation.png"
        style="display: inline-block; margin-top: 10px; padding: 8px 16px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px;"
      >
        Download Result
      </a>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      loading: false,
      error: null,
      resultImage: null,
      nnType: 'unet'
    }
  },
  methods: {
    async handleFileUpload(event) {
      const file = event.target.files[0]
      if (!file) return
      
      if (!file.name.toLowerCase().endsWith('.bmp')) {
        this.error = 'Please upload a BMP file'
        return
      }

      this.loading = true
      this.error = null
      this.resultImage = null

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('nn_type', this.nnType)

        const response = await fetch('http://127.0.0.1:8000/nn_inference', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          throw new Error('Failed to process image')
        }

        const blob = await response.blob()
        this.resultImage = URL.createObjectURL(blob)
      } catch (err) {
        this.error = err.message
      } finally {
        this.loading = false
      }
    }
  }
}
</script>